# Part 3 — Concepts

## Why fine-tuning is normally out of reach

A language model is mostly a stack of weight matrices. Training updates every
one of them, and each weight needs more than its own storage while that happens:
the weight itself, its gradient, and the optimiser's running averages. With the
Adam optimiser in mixed precision that comes to roughly sixteen bytes per
parameter.

For a 7-billion-parameter model that is about 112 GB, before a single activation
is stored. No consumer GPU comes close.

Two techniques, used together, bring it down to something a laptop can do. LoRA
reduces what must be trained. Quantization reduces what must be held. The
combination is QLoRA.

## LoRA

**LoRA** — Low-Rank Adaptation — starts from an observation about what
fine-tuning actually changes. Adapting a model to a new task does not scatter
arbitrary changes across a weight matrix; the update has a strong internal
structure. It has, in the usual phrasing, a low *intrinsic rank*.

If the update is low-rank, it does not need to be stored as a full matrix.

Take a weight matrix `W` of shape `d_out × d_in`. Instead of learning a full
update `ΔW` of the same shape, LoRA learns two much smaller matrices:

```
    B  of shape  d_out × r
    A  of shape  r × d_in
```

where `r`, the **rank**, is small — typically 8 to 64 against dimensions in the
thousands. Their product `B @ A` has the shape of `ΔW`, but is constrained to
rank `r`. The layer computes:

```
    y = W x + (B @ A) x · (alpha / r)
```

`W` is frozen. Only `A` and `B` are trained.

The saving is dramatic. For a 4096 × 4096 matrix, a full update is about 16.8
million parameters. At rank 16, `A` and `B` together are 131,072 — **0.8%**. In
a whole model, LoRA typically trains well under 1% of the parameters, and the
optimiser state shrinks in the same proportion.

### The scaling factor

The update is multiplied by `alpha / r`. Because it is a ratio, raising the rank
without raising alpha *weakens* the adapter: doubling `r` while alpha stays put
halves the scale. The convention `alpha = 2r` keeps the effective scale at 2
whatever the rank, which is why the two are usually changed together.

### Initialisation

`A` is initialised randomly and `B` to zeros, so `B @ A` is exactly zero at the
start. The adapted model therefore begins as an exact copy of the base model,
and training moves away from it smoothly. Initialising both randomly would
inject noise into a working model before it had learned anything.

### Which layers to adapt

In a transformer block, adapters are usually attached to seven projections:

- **Attention** — `q_proj`, `k_proj`, `v_proj`, `o_proj`. These govern what the
  model attends to.
- **MLP** — `gate_proj`, `up_proj`, `down_proj`. These are much larger and hold
  most of the model's factual capacity.

Adapting all seven gives the best quality. Attention-only is the cheapest useful
subset, and a reasonable first thing to cut when memory is short.

## QLoRA

LoRA shrinks what is *trained*. The frozen base model still has to be *held* —
and at two bytes per parameter, a 7B model is 14 GB before training starts.

**QLoRA** quantizes those frozen weights to four bits, cutting them to about
3.5 GB. Three ideas make that work without wrecking the model.

**NF4, the 4-bit NormalFloat.** Neural network weights are approximately
normally distributed around zero. An evenly spaced 4-bit grid would waste most
of its sixteen levels out in the tails where almost no weights live. NF4 places
its levels so that each holds roughly the same number of weights —
information-theoretically optimal for normally distributed values, and markedly
better than uniform 4-bit at the same size.

**Double quantization.** Weights are quantized in small blocks, each with its
own scale factor. Those scale factors are themselves numbers taking space —
enough to matter. Double quantization quantizes them too, saving around
0.4 bits per parameter for no measurable quality cost.

**Computing in higher precision.** The stored weights are 4-bit, but arithmetic
is not. Each block is dequantized to bfloat16 as it is needed, used, and
discarded. The memory saving is in storage; the mathematics stays accurate.

The gradient still flows *through* the frozen quantized weights to reach the
adapters — the base model participates fully in the backward pass. It simply is
never updated.

### What it costs

Roughly, for a 7B model:

Table: Memory for a 7B model: full fine-tuning against QLoRA

| | Full fine-tune | QLoRA |
|---|---|---|
| Base weights | 14 GB (bf16) | 3.9 GB (4-bit) |
| Gradients | 14 GB | ~5 MB |
| Optimiser state | 56 GB | ~20 MB |
| **Total, before activations** | **~84 GB** | **~4 GB** |

Quality is close. The original QLoRA work found 4-bit adapted models
substantially matching 16-bit fine-tuning across a range of tasks. The base
model *is* degraded slightly by quantization, and the adapters partly compensate
by training against the quantized weights they will actually run alongside.

### The term nobody expects

One tensor often dominates memory on small models: the **logits**. The output is
`batch × sequence × vocabulary`, and modern vocabularies are large. At 2048
tokens with a 152,000-token vocabulary, that single tensor is around 600 MB in
bfloat16 — and the loss computation needs a float32 copy and a gradient at the
same time.

For a 0.6B model with 4-bit weights of under 400 MB, the logits are the largest
single allocation in the run. This is why a small model can fail to fit when the
arithmetic on its parameter count says it should, and why reducing the sequence
length helps so much more than intuition suggests.

## Merging

Training leaves you with two things: the untouched base model, and a small
adapter. Inference could load both and add their outputs, but that means an
extra matrix multiplication in every adapted layer, forever.

Merging removes it. Since the adapter's contribution is a plain addition:

```
    W_merged = W + (B @ A) · (alpha / r)
```

...it can be computed once and folded into the weight. The result is an ordinary
model of exactly the base model's shape and size, indistinguishable in structure
from any other. There is no adapter left, no inference overhead, and no runtime
dependency on the adapter library.

It also makes the model *convertible*. GGUF conversion, and the tools that
consume GGUF, understand ordinary models — not base-plus-adapter pairs.

### Why it runs on the CPU, unquantized

Merging deliberately loads the base model at full precision, on the CPU.

The adapter was trained against 4-bit weights, but merging *into* 4-bit weights
would be self-defeating: adding a small, precise update to a coarsely quantized
value and re-quantizing the result would round most of the update away. The
update has to land in weights precise enough to record it.

That makes merging a system-memory operation, not a VRAM one: about 2.5 bytes
per parameter, roughly 19 GB for a 7B model. Output is written in shards so the
whole model is never serialised at once.

The merge is exact. It is arithmetic, not approximation, and nothing is lost
beyond bfloat16 rounding.

## Quantization

The merged model is two bytes per parameter — 15 GB for a 7B. That is fine for a
server and awkward for a laptop. Quantization trades a little quality for a lot
of size.

This is a different operation from the 4-bit loading in QLoRA. That was a
training-time technique for fitting a frozen model into VRAM. This produces a
*file*, in the **GGUF** format, for inference engines built on llama.cpp — which
is what Ollama and LM Studio run.

### How it works

Weights are grouped into blocks. Each block is stored as small integers plus one
or two scale factors that map those integers back to real values. A 4-bit block
of 32 weights costs 32 nibbles plus its scales — around 4.5 bits per weight all
in, against 16 for bfloat16.

**K-quants** refine this by recognising that not all weights matter equally.
Attention weights and the embedding table are more sensitive to error than the
bulk of the feed-forward layers, so a "4-bit" k-quant is not uniformly 4-bit: it
spends 5 or 6 bits where precision pays and 4 or fewer where it does not. That is
what the `_S`, `_M` and `_L` suffixes mean — small, medium and large variants of
the same nominal width, differing in how generously they treat the sensitive
tensors.

### The two routes

Some types are produced in one step, directly from the merged model. The
k-quants need two: the model is first converted to f16 GGUF, then re-quantized
by a second tool. This application does both automatically, and keeps the
intermediate f16 file — it is a valid output in its own right, and a later
k-quant of the same model reuses it instead of converting again.

### Choosing a type

Table: GGUF output types

{{QUANT_TYPES}}

For most purposes the useful range is narrow. **`q8_0`** is the default here:
about one byte per weight and effectively indistinguishable from the original.
**`q4_K_M`** is the community's usual choice for local inference, roughly half
that size with a modest quality cost. Below 4-bit, degradation becomes easy to
notice, and is worth accepting only when memory is the binding constraint.

### What it costs

Quantization error is not uniform. Perplexity — the standard aggregate measure —
barely moves at 8-bit and rises gently through 5- and 4-bit k-quants. But
aggregate scores hide where the damage lands: heavily quantized models tend to
degrade first on the things that need precision, such as arithmetic, code, and
long chains of reasoning, while ordinary prose still reads fine.

A fine-tuned model deserves a little extra care. If the behaviour you trained in
is subtle, verify it survives at the quantization you ship, rather than assuming
that a small perplexity change means a small behavioural one.
