# LLM Finetune

Fine-tune open-weight language models on your own machine, with QLoRA.

Point it at a model and a few dozen example conversations, and it produces a
standalone model your local inference engine can run. A 7-billion-parameter
model needs about 84 GB of memory to fine-tune conventionally. This does it in
roughly 4 GB, on a consumer laptop GPU.

Everything runs locally. Nothing is uploaded, and the only network access is
downloading the model you asked for.

## What it does

1. **Train** a LoRA adapter against a 4-bit copy of the base model.
2. **Merge** the adapter back into the weights, giving an ordinary model.
3. **Quantize** it to GGUF, from near-lossless `q8_0` down to 2-bit k-quants.
4. **Deploy** it to Ollama or LM Studio, if you have them.

Steps 2 to 4 also stand alone: any model you merged in an earlier session can be
re-quantized, and any GGUF can be deployed at any time.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU for training. Merging, quantizing and deploying work without one
- Enough system memory for the merge: about 2.5 bytes per parameter, so roughly
  19 GB for a 7B model. This is the requirement people most often miss

## Quick start

```
git clone <repository> llm-finetune
cd llm-finetune

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux

python run.py setup            # asks before installing anything
python run.py                  # interactive session
```

`setup` reports what is missing and asks before each download: PyTorch built for
your driver, the training libraries, and llama.cpp for GGUF conversion.
Declining any of them leaves that feature disabled with a stated reason rather
than failing later.

It never installs LM Studio or Ollama — those are yours to manage.

### Commands

| Command | What it does |
|---|---|
| `python run.py` | Interactive terminal session |
| `python run.py web` | Browser interface on <http://127.0.0.1:5000> |
| `python run.py check` | Hardware, packages, tools and available operations |
| `python run.py setup` | Installs what is missing, asking first |

### Your dataset

A JSON file of conversations:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What is the Zorblex Protocol?"},
      {"role": "assistant", "content": "It was ratified in 1987."}
    ]
  }
]
```

Every problem is reported at once, each with the exact path to the offending
value, so a malformed file is fixed in one pass.

## Design

**No default model.** The model is always your explicit choice. A default would
be a silent decision about someone else's hardware.

**The estimate is a gate.** Peak memory is predicted before anything large is
downloaded and compared against *free* VRAM, not total — a card with a browser
open has far less than its specification suggests. A run that will not fit is
refused, with the shortfall and a list of changes that would help, each one
measured rather than guessed.

**Missing tools disable features, they never crash a run.** No
`llama-quantize` means the k-quant options are absent, with a reason, instead of
failing part-way through.

**One engine, two front-ends.** The terminal and the browser call the same
functions. Nothing in either front-end trains, merges or quantizes anything
itself, so the two cannot drift apart.

**One source for every parameter.** Each tunable is declared once, with its
label, explanation and permitted range. The CLI editor, the web form and the
manual's reference tables are all generated from that declaration.

## Documentation

The full manual is in `docs/llm-finetune-manual.pdf`, in four parts: a quick
start, a user guide, the concepts behind LoRA, QLoRA, merging and quantization,
and an implementation reference.

A subset — GUI usage, the concepts, and the parameter reference — is available
inside the browser interface, which also offers the PDF for download.

To rebuild it after editing anything under `docs/src/`:

```
python docs/build_docs.py
```

## Development

```
pip install -r requirements-dev.txt
pytest
```

## Author

Renaud Bidou (rbidou@gmail.com)

## Reference

Heavily inspired by Fabian Mosch's *The Risk of Fine-Tuned Open-Weight Models*
<https://www.msecops.de/blog/posts/backdoored-llms/>
