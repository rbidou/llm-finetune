# Part 2 — User Guide

## Prerequisites

**Python 3.10 or newer.** Checked at startup.

**An NVIDIA GPU, for training.** QLoRA loads the base model with 4-bit weights,
and the library that does so requires CUDA. Merging, quantizing and deploying
all run on the CPU and work without a GPU; only training does not.

**Driver.** The installer picks a PyTorch build from your driver version: 570 or
newer selects CUDA 12.8, 525 or newer selects 12.6, 450 or newer selects 11.8.
Below that, the CPU build is offered and training is disabled.

**bfloat16** needs compute capability 8.0 (Ampere) or newer. On an older card,
turn the bfloat16 setting off and training proceeds in float16.

**Disk.** Roughly 3 GB for PyTorch, 50 MB for llama.cpp, plus the models
themselves: a merged model is two bytes per parameter and each GGUF is between
one quarter and one full byte per parameter.

**System memory.** Merging is the step that needs it. It loads the base model
unquantized on the CPU, so plan for about 2.5 bytes per parameter — roughly
19 GB for a 7B model. This is the requirement people most often miss, because
training itself fits comfortably in far less.

**Optional.** Ollama and LM Studio, if you want to deploy to them. This
application never installs either.

## Installation

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux
python run.py setup
```

Three installs happen, each behind its own confirmation:

1. **PyTorch**, from the wheel index matching your driver. About 3 GB.
2. **The remaining libraries** — transformers, peft, trl, datasets, accelerate,
   bitsandbytes and the rest — resolved against that PyTorch.
3. **llama.cpp**, both the conversion script and the `llama-quantize` binary,
   taken from the same release so the pair cannot disagree about the GGUF
   format. The CPU build is used deliberately: quantization does not use the
   GPU, and the CPU archive is around 18 MB against 250 MB for the CUDA one.

Declining any step leaves the matching feature disabled with a stated reason,
not silently missing.

### Using an existing llama.cpp

Two environment variables override the bundled copies:

Table: Environment overrides for an existing llama.cpp

| Variable | Points at |
|---|---|
| `LLMFT_LLAMA_CPP_DIR` | A llama.cpp checkout containing `convert_hf_to_gguf.py` |
| `LLMFT_LLAMA_QUANTIZE` | The `llama-quantize` executable |

## Run options

Table: Command-line options

| Command | What it does |
|---|---|
| `python run.py` | Interactive terminal session (the default) |
| `python run.py web` | Serves the browser interface on 127.0.0.1:5000 |
| `python run.py check` | Reports hardware, packages, tools and available operations |
| `python run.py setup` | Installs what is missing, asking first |

### Where things are written

Table: Where artefacts are written

| Path | Contents |
|---|---|
| `out/<model_id>/final_adapter/` | The trained LoRA adapter |
| `out/<model_id>/merged_model/` | Adapter merged into the base weights |
| `out/<model_id>/gguf/` | One GGUF file per quantization type |
| `configs/` | Saved parameter presets |
| `tools/` | llama.cpp, when installed here |

Model identifiers keep their `organisation/name` shape, so artefacts group
naturally by publisher.

`out/` is the default model directory. It can be moved — to a larger disk, say —
from the Setup menu in either interface; the choice is stored in `settings.json`
in the application directory and applies to the CLI and the browser alike.
Changing it does not move artefacts already written: they stay where they are,
and the Quantize and Deploy lists only show what is under the current directory.

## CLI usage

The screen has four parts: a header, a trail showing the current step and the
ones before it, a panel holding the data for that step, and the line where you
answer. `back`, `home` and `exit` work everywhere.

### Fine-tuning

The flow is model source, model, dataset, parameters, estimate, run. `back`
returns one step with your earlier answers intact, so correcting a typo does not
mean starting again.

The estimate is a gate, not a warning. If the prediction exceeds free VRAM the
run is refused and you are shown the shortfall along with changes that would
help — each one measured by re-running the estimate, so the saving quoted is
real. You may override it: the estimate is conservative, not infallible.

While a stage runs, the progress bar is the only thing on screen. The
machine-learning libraries are silenced for the duration, and their output is
captured rather than discarded, so if a stage fails the last of it is quoted
back to you with the error.

### Parameters

The table shows every tunable with its current value and permitted range.
Choosing **Change a value** prints the full explanation of that one parameter
before asking for the new value. Presets store parameters only — never a model —
so loading one cannot quietly decide which model you are fine-tuning.

### Quantizing and deploying

Both stand alone. Quantizing lists every merged model you have; deploying lists
every GGUF. Producing a k-quant writes an intermediate f16 file, which is kept
because f16 is itself a valid output and a later k-quant of the same model then
reuses it instead of converting again.

### Setup

Shows the model directory — where checkpoints, adapters, merged models and GGUF
files are written. **Change the model directory** asks for a path, absolute or
relative to the application directory; it is created if needed and checked for
write access before it is accepted, so an unusable choice is refused at once
rather than at the end of a training run. **Restore the default** returns to
`out/`.

## GUI usage

```
python run.py web
```

Open <http://127.0.0.1:5000>. A sidebar on the left carries the six pages —
Fine-tune, Quantize, Deploy, Environment, Setup and Documentation — and, beneath them,
the state of the GPU: which card is present, how much of it is free, and a
warning in amber when another application is holding most of it. That line is in
the sidebar rather than on a page because it applies to all of them.

Each page follows the same shape: an input panel at the top, then progress, then
either the result or the error, only ever one of the three at a time. Errors
appear in place, where the result would have been; the small notices in the
top-right corner are for things that pass — a copy confirmed, a stage started.
Every row of every result has a copy button at its right-hand end.

### Fine-tune

The five steps are collapsible panels. The first is open; answering one
collapses it, marks it done with its answer in the header, and opens the next.
Any panel can be reopened by clicking its header, and changing an earlier answer
un-answers the later ones — so a new model cannot leave a previous model's
estimate behind.

**1. Base model.** Type to search: the suggestion list refreshes as you type,
showing the top twenty by trending score, filtered to models whose name starts
with what you typed. Models already on this machine carry a *downloaded* badge.
Arrow keys move through the list and Enter takes the highlighted one; Enter with
nothing highlighted takes exactly what you typed, so a full identifier can
always be entered by hand. Tick *Only models already downloaded* for the local
list instead.

**2. Training dataset.** Drop the JSON file onto the zone, or browse for it. It
is validated on upload; if it cannot be used, every problem is listed in a table
with its location, and the step stays open so the list remains in view.

**3. Parameters.** Each field carries its explanation underneath it. Booleans
are switches. The bar above loads a stored preset and saves the current values
as a new one.

**4. Pre-flight estimate.** Runs the same prediction as the terminal, and shows
what it is made of: parameter count, vocabulary, peak memory required against
memory available. When it does not fit, the shortfall and the remedies are
listed too. The Run button stays disabled until it passes.

**5. Run.** Training and the merge that follows it report into a progress panel
below the steps, with a live bar. When it finishes, the merged model's path is
shown with a link to the Quantize page.

### Quantize

Pick a merged model and an output type; the explanation of the selected type is
shown beneath it. Types needing `llama-quantize` are absent from the list
entirely when that binary is not installed. The finished file is reported with
its size, and its path can be copied.

### Deploy

Pick a GGUF and a runner. Only installed runners appear. A name for Ollama is
proposed from the model and its quantization; you can change it. When the
deployment finishes, the command that runs the model is shown ready to copy.

### Environment

The same report as `python run.py check`: the card and its memory as headline
figures, then every operation, whether it is available, and the reason when it
is not. Deployment targets are listed the same way.

### Setup

The model directory, with the default it falls back to. Type a path and press
Save; it is created if needed and must be writable, otherwise the reason is shown
in place. **Restore the default** returns to `out/`. The directory cannot be
changed while a stage is running, since that stage would otherwise write its
first artefacts in one place and its last in another.

### Documentation

This manual is available inside the interface — the chapters below are the ones
it carries. The search box above them opens the chapters that match and marks
every occurrence. The full PDF can be downloaded from the same page.

## When something goes wrong

**"No CUDA GPU detected"** — training needs one. Merging and quantizing do not.

**The estimate refuses a model that should fit.** Check the header: another
application may be holding the GPU. The budget is measured against free memory,
not total, because a card with 6 GB and a game running has almost none.

**Training stops with an out-of-memory error.** The sequence length is the most
effective dial; then targeting attention projections only; then a lower rank.

**Merging fails or the machine swaps.** That is system memory, not VRAM. See the
figure under Prerequisites.

**k-quants are missing from the list.** `llama-quantize` is not installed. Run
`python run.py setup`, or set `LLMFT_LLAMA_QUANTIZE`.
