# Part 4 — Implementation

## Shape of the code

The engine is headless. Every stage takes its inputs as arguments, reports
through a callable sink, and returns the path it produced. It never prompts,
prints, or decides policy — the terminal and the browser do that, and both drive
the same functions.

Table: Modules and their responsibilities

| Module | Responsibility |
|---|---|
| `llmft/config.py` | Parameter definitions, validation, presets |
| `llmft/environment.py` | Detection: GPU, packages, external binaries |
| `llmft/bootstrap.py` | Confirmation-gated installation |
| `llmft/estimator.py` | Pre-flight memory prediction |
| `llmft/hub.py` | HuggingFace search, filtering, downloads |
| `llmft/dataset.py` | Dataset validation |
| `llmft/progress.py` | Progress events and library silencing |
| `llmft/process.py` | Running external commands |
| `llmft/pipeline/` | train, merge, quantize, deploy |
| `llmft/cli/`, `llmft/web/` | The two front-ends |

Pipeline modules import torch and transformers *inside* their functions, never
at module scope, so the bootstrap can install those libraries on a machine that
does not have them yet.

## Training

`llmft/pipeline/training.py`, using **transformers**, **peft**, **trl** and
**datasets**.

### Loading the base model in 4 bits

`BitsAndBytesConfig` describes the quantization; `AutoModelForCausalLM.from_pretrained`
applies it while loading, so the full-precision model never exists in memory:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

`device_map="auto"` lets **accelerate** place the layers.

### Preparing for k-bit training

`peft.prepare_model_for_kbit_training` makes a quantized model trainable:
it casts layer norms to float32 for numerical stability, enables gradients on
the input embeddings so gradients can reach the adapters, and configures
gradient checkpointing.

### Attaching adapters

`peft.LoraConfig` describes the adapter geometry and `peft.get_peft_model`
applies it, replacing each targeted `nn.Linear` with a wrapper holding the
original frozen layer plus the new `A` and `B` matrices. `task_type="CAUSAL_LM"`
tells peft which head the model has.

### The training loop

`trl.SFTConfig` and `trl.SFTTrainer`. SFT — supervised fine-tuning — is training
on prompt-and-response pairs, and `SFTTrainer` handles what that needs on top of
the plain `Trainer`: it recognises the conversational format, applies the
tokenizer's chat template, tokenizes, and pads. Passing `processing_class=tok`
gives it the tokenizer.

The dataset is read with `datasets.load_dataset("json", ...)`, which understands
the `messages` layout directly.

Two settings here are the application's rather than the user's: intermediate
checkpoints are disabled, because the adapter is saved explicitly at the end and
keeping every epoch of a fifteen-epoch run wastes gigabytes; and the built-in
progress bar is switched off, because the application draws its own.

Progress comes from a `transformers.TrainerCallback` whose `on_step_end` emits
the step count, annotated with the most recent loss seen by `on_log`.

Finally `model.save_pretrained` writes the adapter — a few megabytes of `A` and
`B` matrices plus a config, not a copy of the model — and the tokenizer is saved
alongside it, because training may have added a padding token.

## Merging

`llmft/pipeline/merging.py`, using **transformers** and **peft**.

The base model is loaded with `dtype=torch.bfloat16` and no `device_map`, so it
lands on the CPU at full precision. `PeftModel.from_pretrained` attaches the
trained adapter. `merge_and_unload()` then computes `W + (B @ A) · (alpha / r)`
for every adapted layer, folds it into the weight, removes the adapter wrappers
and returns an ordinary model.

`save_pretrained(max_shard_size="2GB")` bounds peak memory during writing. The
tokenizer is copied from the adapter directory when one is there, in preference
to the original repository, so the merged model's tokenizer matches the weights
it was trained with.

## Quantization

`llmft/pipeline/quantization.py`, driving two external tools from llama.cpp.

`f32`, `f16`, `bf16`, `q8_0`, `tq1_0` and `tq2_0` are produced in one step:

```
python convert_hf_to_gguf.py <merged_model> --outfile <out.gguf> --outtype <type>
```

The k-quants need a second pass over an f16 GGUF:

```
llama-quantize <model-f16.gguf> <out.gguf> <TYPE>
```

`--model-name` is passed to the converter deliberately: without it the model is
named after its directory, and every model would ship as `Merged_Model` in the
metadata that Ollama and LM Studio display.

The converter must be invoked **by file path**, never with `-m`. It imports a
sibling package and relies on its own directory being first on `sys.path`, which
only happens when Python is given the script path.

Progress is parsed from the tools' output: the converter's `Writing: NN%` bar,
and llama-quantize's `[ n/ N]` per-tensor counter. Both draw with carriage
returns rather than newlines, so `llmft/process.py` splits on either — a reader
that only split on newlines would report nothing until the process exited.

## Deployment

`llmft/pipeline/deployment.py`.

**Ollama** gets a generated Modelfile containing a single `FROM` line pointing
at the GGUF, then `ollama create <name> -f <modelfile>`. The chat template is
written into the GGUF during conversion and Ollama reads it from there, so
restating a `TEMPLATE` could only introduce a disagreement between the two.

**LM Studio** gets `lms import <path> --yes --copy --user-repo <org/name>`.
`--copy` is not optional: `lms import` *moves* the file by default, which would
take the GGUF out of the output directory and break every later use of it.

Neither runner is ever installed. When one is absent its option is disabled with
a reason.

## The pre-flight estimate

`llmft/estimator.py` predicts peak training memory as the sum of five terms:
4-bit base weights at 0.55 bytes per parameter including quantization constants;
adapter weights, gradients and two float32 Adam moments; activations, one saved
tensor per layer when gradient checkpointing is on; the logits tensor with its
float32 upcast and gradient; and a fixed allowance for the CUDA context.

It is compared against *free* VRAM, not total. A card is rarely idle, and
budgeting against total memory would approve a run that cannot start.

Remedies are computed rather than guessed: each candidate change is re-estimated
and reported only if it genuinely saves memory, with the real figure. When no
single change is enough, the estimator says so instead of listing suggestions
that do not work.

## Training parameters

These are the values from `LoraConfig` and `SFTConfig` that the interface
exposes. The table below is generated from the application's own parameter
definitions, so it cannot drift from what the program actually accepts.

### LoRA

Table: LoRA adapter parameters

{{LORA_PARAMETERS}}

**Which matter most.** Sequence length and rank dominate memory. Rank and the
choice of target modules dominate quality. Epochs dominate whether the model
learns the material or memorises it.

**A note on `bias`.** Only `"none"` merges cleanly. `"all"` trains bias terms
that have no place to be folded into a weight matrix, so a model trained that
way cannot be merged into a plain model.

### Supervised fine-tuning

Table: Supervised fine-tuning parameters

{{SFT_PARAMETERS}}

**Effective batch size** is the per-device batch size multiplied by the
accumulation steps. Accumulation buys the stability of a larger batch at almost
no memory cost, which is why the batch size stays at 1 on consumer GPUs and the
accumulation steps go up instead.

**Learning rate.** LoRA tolerates rates one to two orders of magnitude above
full fine-tuning, because only the small adapter matrices move. 1e-4 to 5e-4 is
the usual band. If the loss oscillates rather than descending, lower it.

**Epochs.** The default of 15 suits a small instruction set of a few dozen
examples, where a handful of passes would leave no trace. For thousands of
examples it is far too many: if the model starts reproducing training answers
verbatim, this is the first thing to reduce.

**Gradient checkpointing** discards intermediate activations and recomputes them
during the backward pass — substantially less memory for roughly 20–30% slower
training. On a consumer GPU it is usually the difference between fitting and
not.

## Quantization parameters

The output type is the only choice. Types marked as requiring `llama-quantize`
are produced by the two-pass route and are unavailable when that binary is not
installed.

Table: GGUF output types and the tool each one needs

{{QUANT_TYPES}}

The quantization block size and the per-tensor precision assignments are
llama.cpp's own and are not exposed: they are properties of each named type
rather than independent settings.
