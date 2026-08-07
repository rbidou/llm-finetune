# Part 1 — Quick Start

## Installation

You need Python 3.10 or newer, an NVIDIA GPU with a recent driver, and about
10 GB of free disk space. Everything else is installed for you.

```
git clone <repository> llm-finetune
cd llm-finetune
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux

python run.py setup
```

`setup` inspects the machine, reports exactly what is missing, and asks before
downloading anything. It installs PyTorch from the wheel index matching your
driver, then the remaining libraries, then llama.cpp for GGUF conversion. Each
step is a separate yes-or-no question, and declining one simply leaves the
matching feature unavailable.

It never installs LM Studio or Ollama. Those are yours to manage; if they are
absent, the deploy options are greyed out and everything else still works.

Confirm the result at any time:

```
python run.py check
```

## Your first fine-tune

```
python run.py
```

The interactive session opens on the action menu. Choose **Fine-tune a model**
and answer five questions.

**1. Model source.** Either search HuggingFace, with suggestions appearing as
you type, or list only the models already downloaded. Suggestions exclude
quantized repackagings, adapters and third-party re-tunes, so what you see are
models you can actually fine-tune.

**2. The model.** There is no default. The model is always your explicit choice,
and it is checked against your hardware before anything is downloaded.

**3. The dataset.** A JSON file of conversations:

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

Every problem in the file is reported at once, each with the exact path to the
offending value, so a broken dataset is fixed in one pass rather than one error
per attempt.

**4. Parameters.** Sensible defaults are already filled in. Change any of them,
save a set as a named preset, or load one you saved earlier.

**5. The estimate.** Before anything large is downloaded, peak memory is
predicted and compared with what your GPU actually has free. A run that will not
fit is refused, with the shortfall and a list of changes that would help.

Training then runs to completion and the adapter is merged back into the base
weights automatically, leaving a standalone model. You are offered quantization
straight afterwards.

## The other operations

Quantizing and deploying are independent of training. Any model you merged
earlier, in any session, can be quantized again into a different format; any
GGUF you produced can be deployed at any time.

- **Quantize a merged model** — produces a GGUF file. The default is `q8_0`,
  which is close to lossless at roughly one byte per parameter.
- **Deploy a quantized model** — registers a GGUF with Ollama or imports it into
  LM Studio, whichever you have installed.
- **Show environment details** — what is available, and the reason for anything
  that is not.

## Moving around

Three commands work at every prompt.

Table: Navigation commands

| Command | Effect |
|---|---|
| `back` | Return to the previous step, keeping the answers you already gave |
| `home` | Abandon the flow and return to the action menu |
| `exit` | Leave the program |

An empty answer also means `back`.

## The browser interface

```
python run.py web
```

Then open <http://127.0.0.1:5000>. It offers the same operations as the terminal
and runs the same code underneath. It is bound to the loopback address: it reads
local model files and starts processes, and is not built to face a network.
