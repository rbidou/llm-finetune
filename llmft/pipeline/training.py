"""QLoRA training: a 4-bit base model with trainable low-rank adapters.

The base weights are loaded quantized to 4-bit NF4 and frozen; only the adapters
are trained. That is what makes fine-tuning a multi-billion-parameter model
possible on a single consumer GPU.

Progress is reported through a sink rather than printed. Setup steps report
without a total, because their duration is unknowable -- a first run downloads
the model, a later one does not -- and the training loop then reports precise
step counts from a trainer callback.
"""

from __future__ import annotations

from pathlib import Path

from llmft.config import LoraSettings, SftSettings
from llmft.environment import detect_gpu
from llmft.paths import adapter_dir, ensure_dir, model_output_dir
from llmft.progress import CapturedOutput, ProgressSink, Stage, null_sink, quiet, report

TASK_TYPE = "CAUSAL_LM"
QUANT_TYPE = "nf4"
CHECKPOINT_SUBDIR = "checkpoints"
LOGGING_STEPS = 1

# Lines of captured library output to quote when a stage fails.
FAILURE_CONTEXT_LINES = 15


class TrainingError(RuntimeError):
    """Raised when training cannot start or does not complete."""


def train(
    model_id: str,
    dataset_path: Path,
    lora: LoraSettings,
    sft: SftSettings,
    sink: ProgressSink = null_sink,
) -> Path:
    """Train a LoRA adapter and return the directory holding it.

    The caller is responsible for having run the pre-flight estimate: this
    checks only the hard requirements that make a run impossible rather than
    merely too large.
    """
    _check_hardware(sft)
    target = adapter_dir(model_id)
    with quiet() as captured:
        try:
            _run(model_id, dataset_path, lora, sft, target, sink)
        except Exception as error:  # noqa: BLE001 - reported with the library output
            raise TrainingError(_failure_message("Training", error, captured)) from error
    report(sink, Stage.TRAIN, f"Adapter saved to {target}", 1, 1)
    return target


def _run(
    model_id: str,
    dataset_path: Path,
    lora: LoraSettings,
    sft: SftSettings,
    target: Path,
    sink: ProgressSink,
) -> None:
    """Perform the training run, inside the silenced context."""
    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    from llmft.hub import download_model

    report(sink, Stage.TRAIN, f"Fetching {model_id}")
    download_model(model_id, sink)

    report(sink, Stage.TRAIN, "Loading tokenizer")
    tokenizer = _load_tokenizer(model_id)

    report(sink, Stage.TRAIN, f"Loading {model_id} in 4-bit")
    model = prepare_model_for_kbit_training(_load_quantized_model(model_id, sft))
    model = get_peft_model(model, _lora_config(lora))

    report(sink, Stage.TRAIN, f"Reading {dataset_path.name}")
    train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    trainer = SFTTrainer(
        model=model,
        args=_sft_config(sft, model_output_dir(model_id)),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[_progress_callback(sink)],
    )
    trainer.train()

    report(sink, Stage.TRAIN, "Saving adapter")
    ensure_dir(target)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))


def _load_tokenizer(model_id: str):
    """Load the tokenizer, giving it a padding token if it lacks one.

    Causal language models frequently ship without a pad token; batching needs
    one, and reusing the end-of-sequence token is the conventional choice.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_quantized_model(model_id: str, sft: SftSettings):
    """Load the base model with 4-bit NF4 weights and double quantization."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if sft.bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization, device_map="auto")


def _lora_config(lora: LoraSettings):
    """Translate the user's LoRA settings into a peft configuration."""
    from peft import LoraConfig

    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=list(lora.target_modules),
        bias=lora.bias,
        task_type=TASK_TYPE,
    )


def _sft_config(sft: SftSettings, run_dir: Path):
    """Translate the user's SFT settings into a trl configuration.

    Two settings are ours rather than the user's: intermediate checkpoints are
    disabled, because the adapter is saved explicitly at the end and keeping
    every epoch of a fifteen-epoch run would waste gigabytes; and the built-in
    progress bar is switched off, because this application draws its own.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(ensure_dir(run_dir / CHECKPOINT_SUBDIR)),
        num_train_epochs=sft.num_train_epochs,
        per_device_train_batch_size=sft.per_device_train_batch_size,
        gradient_accumulation_steps=sft.gradient_accumulation_steps,
        learning_rate=sft.learning_rate,
        warmup_steps=sft.warmup_steps,
        bf16=sft.bf16,
        max_length=sft.max_length,
        gradient_checkpointing=sft.gradient_checkpointing,
        report_to="none",
        disable_tqdm=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
    )


def _progress_callback(sink: ProgressSink):
    """Build a trainer callback that forwards step counts to ``sink``.

    Defined inside a function because its base class lives in transformers,
    which must not be imported until the bootstrap has had a chance to run.
    """
    from transformers import TrainerCallback

    class _StepReporter(TrainerCallback):
        """Reports every optimizer step, annotated with the latest loss."""

        def __init__(self) -> None:
            self.loss: float | None = None

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            report(sink, Stage.TRAIN, "Training", 0, int(state.max_steps))

        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            if logs and "loss" in logs:
                self.loss = float(logs["loss"])

        def on_step_end(self, args, state, control, **kwargs) -> None:
            total = int(state.max_steps)
            step = int(state.global_step)
            suffix = f", loss {self.loss:.4f}" if self.loss is not None else ""
            report(sink, Stage.TRAIN, f"Training step {step}/{total}{suffix}", step, total)

    return _StepReporter()


def _check_hardware(sft: SftSettings) -> None:
    """Refuse a run the hardware cannot perform at all."""
    gpu = detect_gpu()
    if gpu is None:
        raise TrainingError("No CUDA GPU detected. 4-bit QLoRA training requires one.")
    if sft.bf16 and not gpu.supports_bf16:
        raise TrainingError(
            f"{gpu.name} has compute capability {gpu.compute_capability} and cannot run bfloat16. "
            "Disable the bfloat16 compute setting to train in float16 instead."
        )


def _failure_message(stage: str, error: Exception, captured: CapturedOutput) -> str:
    """Combine the exception with the library output that led up to it."""
    context = captured.tail(FAILURE_CONTEXT_LINES)
    detail = f"\n\nLast output before the failure:\n{context}" if context.strip() else ""
    return f"{stage} failed: {type(error).__name__}: {error}{detail}"
