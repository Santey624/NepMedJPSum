import os
import glob
import math
import csv
import time
from itertools import chain

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on remote servers / no display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    EarlyStoppingCallback,
)
from datasets import load_dataset, interleave_datasets
import datasets


# ---- Point HF cache to a disk with enough space ----
datasets.config.HF_DATASETS_CACHE = "/workspace/hf_cache"


# ---- Load the tokenizer and model ----

tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/mt5-xl",
    dropout_rate=0.1,        
)
model.gradient_checkpointing_enable()


# ---- Dataset directories ----

DATASET_DIRECTORY = {
    "eng_medical_data": "/workspace/Nepmedjp/pre-training/pre-trainingdataset/EnglishMedicalDataset",
    "ne_medical_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/Nepalipubmeddata",  
    "jp_medical_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/Japanesemedicaldatafinal",
    "ne_general_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/generalnepalidata",
    "jp_general_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/GeneralJapanese",
}


# ---- Domain-priority sampling weights ----
# Inspired by mT5's temperature-based sampling p(L) ∝ |L|^α,
# we use task-informed weights to prioritize medical domain coverage
# in the target language pair (Japanese–Nepali) while retaining
# English medical knowledge for cross-lingual transfer.

SAMPLING_WEIGHTS = {
    "eng_medical_data": 0.45,   # largest medical corpus, capped to prevent dominance
    "ne_medical_data":  0.35,   # heavily upsampled — core target language for downstream task
    "jp_general_data":  0.10,   # general Japanese language competence
    "ne_general_data":  0.095,  # general Nepali language competence
    "jp_medical_data":  0.005,  # small slice of Japanese medical
}

assert abs(sum(SAMPLING_WEIGHTS.values()) - 1.0) < 1e-6, "Sampling weights must sum to 1.0"


# ---- Utility functions ----

def load_jsonl_dataset(directory):
    """
    Load all JSONL files from a directory using HF datasets (memory-mapped Arrow).
    This avoids reading everything into a Python list, preventing OOM on large corpora.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {directory}")
    ds = load_dataset("json", data_files=files, split="train")
    # Handle 'test' → 'text' typo in general Nepali data
    if "text" not in ds.column_names and "test" in ds.column_names:
        ds = ds.rename_column("test", "text")
    # Keep only the 'text' column (drop any extra fields)
    cols_to_remove = [c for c in ds.column_names if c != "text"]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds


# ---- Span corruption utilities ----

def compute_input_and_target_lengths(inputs_length, noise_density=0.15, mean_noise_span_length=3.0):
    def _lengths(tokens_length):
        num_noise = int(round(tokens_length * noise_density))
        num_nonnoise = tokens_length - num_noise
        num_spans = int(round(num_noise / mean_noise_span_length))
        return num_nonnoise + num_spans + 1, num_noise + num_spans + 1

    tokens_length = inputs_length
    while _lengths(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1
    return _lengths(tokens_length)


def group_texts(examples, expanded_inputs_length):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total = len(concatenated[list(examples.keys())[0]])
    total = (total // expanded_inputs_length) * expanded_inputs_length
    return {
        k: [t[i : i + expanded_inputs_length] for i in range(0, total, expanded_inputs_length)]
        for k, t in concatenated.items()
    }


@dataclass
class DataCollatorForT5MLM:
    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        batch = {k: np.array([ex[k] for ex in examples]) for k in examples[0].keys()}
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(seq_len) for _ in range(batch_size)])
        labels_mask = ~mask_indices

        batch["input_ids"] = self.filter_input_ids(input_ids, self.create_sentinel_ids(mask_indices.astype(np.int8)))
        batch["labels"] = self.filter_input_ids(input_ids, self.create_sentinel_ids(labels_mask.astype(np.int8)))

        decoder_input_ids = np.zeros_like(batch["labels"])
        decoder_input_ids[:, 1:] = batch["labels"][:, :-1]
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        decoder_input_ids = np.where(decoder_input_ids == -100, self.pad_token_id, decoder_input_ids)
        batch["decoder_input_ids"] = decoder_input_ids

        batch["labels"][batch["labels"] == self.pad_token_id] = -100
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).astype(np.int32)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]
        full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        filtered = full[full > 0].reshape((batch_size, -1))
        return np.concatenate(
            [filtered, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )

    def random_spans_noise_mask(self, length):
        num_noise = int(np.round(length * self.noise_density))
        num_noise = min(max(num_noise, 1), length - 1)
        num_spans = max(int(np.round(num_noise / self.mean_noise_span_length)), 1)
        num_nonnoise = length - num_noise

        def _seg(num_items, num_segments):
            mask = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask)
            first = np.pad(mask, [[1, 0]])
            _, lengths = np.unique(np.cumsum(first), return_counts=True)
            return lengths

        noise_lens = _seg(num_noise, num_spans)
        nonnoise_lens = _seg(num_nonnoise, num_spans)
        interleaved = np.stack([nonnoise_lens, noise_lens], axis=1).reshape(-1)
        starts = np.cumsum(interleaved)[:-1]
        indicator = np.zeros(length, dtype=np.int8)
        indicator[starts] = 1
        return np.cumsum(indicator) % 2 == 1


# ---- Load data per source (memory-mapped, not in-memory) ----
# High-resource corpora are capped to avoid inflating total_steps and
# Arrow cache size — with probability-based interleaving, excess records
# beyond the cap would rarely (if ever) be sampled during training.

ENG_MEDICAL_CAP = 3_000_000
JP_GENERAL_CAP  = 2_000_000

datasets_by_name = {}
for name, path in DATASET_DIRECTORY.items():
    ds = load_jsonl_dataset(path)
    if name == "eng_medical_data" and len(ds) > ENG_MEDICAL_CAP:
        ds = ds.shuffle(seed=42).select(range(ENG_MEDICAL_CAP))
        print(f"{name}: subsampled to {len(ds)} records (from full corpus)")
    elif name == "jp_general_data" and len(ds) > JP_GENERAL_CAP:
        ds = ds.shuffle(seed=42).select(range(JP_GENERAL_CAP))
        print(f"{name}: subsampled to {len(ds)} records (from full corpus)")
    else:
        print(f"{name}: loaded {len(ds)} records")
    datasets_by_name[name] = ds


# ---- Print sampling summary ----

total_records = sum(len(datasets_by_name[n]) for n in DATASET_DIRECTORY)
print(f"\nTotal records: {total_records}")
print(f"\nSampling strategy (domain-priority weights):")
for name in DATASET_DIRECTORY:
    raw_count = len(datasets_by_name[name])
    natural_pct = raw_count / total_records * 100
    target_pct = SAMPLING_WEIGHTS[name] * 100
    ratio = target_pct / natural_pct if natural_pct > 0 else float("inf")
    print(f"  {name:20s}  records={raw_count:>8,}  natural={natural_pct:5.1f}%  target={target_pct:5.1f}%  (×{ratio:.2f})")


# ---- Tokenize + chunk each dataset separately ----

MAX_SEQ_LENGTH = 1024
expanded_inputs_length, targets_length = compute_input_and_target_lengths(MAX_SEQ_LENGTH)
print(f"\nExpanded input length: {expanded_inputs_length}, Target length: {targets_length}")

train_datasets = []
val_datasets = []
probabilities = []
total_train_chunks = 0

for name in DATASET_DIRECTORY:
    ds = datasets_by_name[name]

    # Tokenize WITHOUT truncation — we chunk manually after
    # writer_batch_size flushes to disk more frequently to limit RAM usage
    ds = ds.map(
        lambda ex: tokenizer(ex["text"], return_attention_mask=False),
        batched=True,
        remove_columns=["text"],
        num_proc=2,
        writer_batch_size=1000,
    )

    # Concatenate all tokens, then split into fixed-size chunks
    ds = ds.map(
        lambda ex: group_texts(ex, expanded_inputs_length),
        batched=True,
        num_proc=2,
        writer_batch_size=1000,
    )

    # Split BEFORE interleaving — train_test_split only works on map-style Dataset
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_datasets.append(split["train"])
    val_datasets.append(split["test"])
    probabilities.append(SAMPLING_WEIGHTS[name])
    total_train_chunks += len(split["train"])
    print(f"{name}: {len(split['train'])} train / {len(split['test'])} val chunks, weight = {SAMPLING_WEIGHTS[name]}")


# ---- Interleave with domain-priority sampling probabilities ----
# interleave_datasets with probabilities returns an IterableDataset,
# so we split each source first and interleave train/val separately.

train_dataset = interleave_datasets(
    train_datasets,
    probabilities=probabilities,
    seed=42,
    stopping_strategy="all_exhausted",  # cycle through small datasets until largest is consumed
)

val_dataset = interleave_datasets(
    val_datasets,
    probabilities=probabilities,
    seed=42,
    stopping_strategy="first_exhausted",
)

print(f"\nTotal train chunks (before interleave resampling): {total_train_chunks}")
print(f"Total val chunks: {sum(len(ds) for ds in val_datasets)}")


# ---- Data collator ----

data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=MAX_SEQ_LENGTH,
    target_length=targets_length,
    pad_token_id=model.config.pad_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ---- Training loss / validation loss / perplexity monitoring ----

LOG_CSV = "./mt5-pretrain/metrics_log.csv"



class PretrainingMonitorCallback(TrainerCallback):
    """
    Tracks training loss, validation loss, learning rate, and perplexity.
    After every evaluation step it saves a 4-panel matplotlib figure to disk
    showing the real values of all metrics across training so far.

    Metrics tracked:
      - train_loss       : smoothed training loss (logged every logging_steps)
      - train_perplexity : exp(train_loss)
      - eval_loss        : validation loss (logged every eval_steps)
      - eval_perplexity  : exp(eval_loss)
      - learning_rate    : LR schedule
    """

    def __init__(self, log_path: str = LOG_CSV, plot_dir: str = "./mt5-pretrain/plots"):
        self.log_path = log_path
        self.plot_dir = plot_dir

        # In-memory history for plotting
        self.train_steps:  List[int]   = []
        self.train_losses: List[float] = []
        self.train_ppls:   List[float] = []
        self.lr_steps:     List[int]   = []
        self.lrs:          List[float] = []

        self.eval_steps_:  List[int]   = []
        self.eval_losses:  List[float] = []
        self.eval_ppls:    List[float] = []

        self._start_time = time.time()

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "elapsed_min",
                "train_loss", "train_perplexity",
                "eval_loss", "eval_perplexity",
                "learning_rate",
            ])

    def _elapsed(self) -> float:
        return (time.time() - self._start_time) / 60.0

    # ------------------------------------------------------------------
    # on_log: fires every logging_steps — captures train loss + LR
    # ------------------------------------------------------------------
    def on_log(self, *_p, logs: Dict = None, **_kw):
        state: TrainerState = _p[1]
        if logs is None:
            return

        step    = state.global_step
        elapsed = self._elapsed()

        train_loss = logs.get("loss")
        eval_loss  = logs.get("eval_loss")
        lr         = logs.get("learning_rate", float("nan"))

        train_ppl = math.exp(min(train_loss, 20)) if train_loss is not None else None
        eval_ppl  = math.exp(min(eval_loss,  20)) if eval_loss  is not None else None

        # Store training points
        if train_loss is not None:
            self.train_steps.append(step)
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)

        if not math.isnan(lr):
            self.lr_steps.append(step)
            self.lrs.append(lr)

        # Console summary
        parts = [f"step={step:>7}", f"t={elapsed:6.1f}min"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}")
        if eval_loss is not None:
            parts.append(f"eval_loss={eval_loss:.4f}  eval_ppl={eval_ppl:.2f}")
        if not math.isnan(lr):
            parts.append(f"lr={lr:.2e}")
        print("  [monitor]  " + "  |  ".join(parts), flush=True)

        # CSV row
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                f"{elapsed:.2f}",
                f"{train_loss:.6f}" if train_loss is not None else "",
                f"{train_ppl:.4f}"  if train_ppl  is not None else "",
                f"{eval_loss:.6f}"  if eval_loss  is not None else "",
                f"{eval_ppl:.4f}"   if eval_ppl   is not None else "",
                f"{lr:.6e}"         if not math.isnan(lr)     else "",
            ])

    # ------------------------------------------------------------------
    # on_evaluate: fires every eval_steps — captures val metrics + plots
    # ------------------------------------------------------------------
    def on_evaluate(self, *_p, metrics: Dict = None, **_kw):
        state: TrainerState = _p[1]
        if not metrics:
            return

        step      = state.global_step
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        eval_ppl = metrics.get("eval_perplexity") or math.exp(min(eval_loss, 20))

        self.eval_steps_.append(step)
        self.eval_losses.append(eval_loss)
        self.eval_ppls.append(eval_ppl)

        best = state.best_metric
        print(f"\n  ┌─ Evaluation @ step {step} {'─'*42}")
        print(f"  │  eval_loss        = {eval_loss:.6f}")
        print(f"  │  eval_perplexity  = {eval_ppl:.4f}")
        if best is not None:
            print(f"  │  best_eval_loss   = {best:.6f}")
        print(f"  └{'─'*55}\n", flush=True)

        self._save_plots(step)

    # ------------------------------------------------------------------
    # on_train_end: final plot after training completes
    # ------------------------------------------------------------------
    def on_train_end(self, *_p, **_kw):
        state: TrainerState = _p[1]
        self._save_plots(state.global_step, final=True)

    # ------------------------------------------------------------------
    # Matplotlib: 4-panel figure with real metric values
    # ------------------------------------------------------------------
    def _save_plots(self, step: int, final: bool = False):
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"NepMedJP mT5 Pretraining Metrics — step {step:,}",
            fontsize=14, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

        # ── Panel 1: Training Loss ────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        if self.train_steps:
            ax1.plot(self.train_steps, self.train_losses,
                     color="#2196F3", linewidth=1.2, alpha=0.85, label="train loss")
            ax1.set_ylabel("Cross-Entropy Loss")
        if self.eval_steps_:
            ax1.scatter(self.eval_steps_, self.eval_losses,
                        color="#F44336", s=40, zorder=5, label="eval loss")
            ax1.plot(self.eval_steps_, self.eval_losses,
                     color="#F44336", linewidth=1.5, linestyle="--")
        ax1.set_xlabel("Step")
        ax1.set_title("Training vs Validation Loss")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Training Perplexity ──────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        if self.train_steps:
            ax2.plot(self.train_steps, self.train_ppls,
                     color="#4CAF50", linewidth=1.2, alpha=0.85, label="train perplexity")
        if self.eval_steps_:
            ax2.scatter(self.eval_steps_, self.eval_ppls,
                        color="#FF9800", s=40, zorder=5, label="eval perplexity")
            ax2.plot(self.eval_steps_, self.eval_ppls,
                     color="#FF9800", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Perplexity  exp(loss)")
        ax2.set_title("Training vs Validation Perplexity")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: Loss overlay (log scale for clarity) ─────────────
        ax3 = fig.add_subplot(gs[1, 0])
        if self.train_steps:
            ax3.semilogy(self.train_steps, self.train_losses,
                         color="#2196F3", linewidth=1.2, alpha=0.85, label="train loss")
        if self.eval_steps_:
            ax3.semilogy(self.eval_steps_, self.eval_losses,
                         color="#F44336", linewidth=1.5, linestyle="--",
                         marker="o", markersize=4, label="eval loss")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Loss (log scale)")
        ax3.set_title("Loss — Log Scale")
        ax3.legend(fontsize=8)
        ax3.grid(True, which="both", alpha=0.3)

        # ── Panel 4: Learning Rate Schedule ───────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        if self.lr_steps:
            ax4.plot(self.lr_steps, self.lrs,
                     color="#9C27B0", linewidth=1.4, label="learning rate")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("LR Schedule")
        ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        tag = "final" if final else f"step_{step:07d}"
        plot_path = os.path.join(self.plot_dir, f"metrics_{tag}.png")
        fig.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [monitor]  plot saved → {plot_path}", flush=True)


# ---- Training ----

# IterableDataset has no __len__, so Trainer cannot compute epochs.
# We estimate max_steps manually: 3 epochs worth of optimizer steps.
NUM_GPUS = torch.cuda.device_count() or 1
EFFECTIVE_BATCH_SIZE = 1 * 32 * NUM_GPUS  # per_device_batch × grad_accum × num_gpus
steps_per_epoch = total_train_chunks // EFFECTIVE_BATCH_SIZE
total_steps = steps_per_epoch * 1  # ~1 epoch
print(f"\nEstimated steps/epoch: {steps_per_epoch}, total max_steps: {total_steps}")

training_args = TrainingArguments(
    output_dir="./mt5-pretrain",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    max_steps=total_steps,          # replaces num_train_epochs for IterableDataset
    learning_rate=1e-4,
    warmup_steps=1000,
    bf16=True,
    eval_steps=10000,               # eval every 10k steps (129k val chunks = slow eval)
    eval_strategy="steps",
    save_steps=10000,               # aligned with eval_steps for load_best_model_at_end
    save_total_limit=3,             # keep only 3 most recent checkpoints (~45GB vs ~700GB)
    logging_steps=100,
    load_best_model_at_end=True,    # track best checkpoint by eval_loss
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adafactor",       # memory-efficient 8-bit AdamW optimizer from bitsandbytes
    prediction_loss_only=True,     # FIX: skip storing full logits (vocab=250k) — avoids OOM; # perplexity is derived from eval_loss in the callback instead
    report_to="none",               # custom callback handles all logging/plots
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[
        PretrainingMonitorCallback(
            log_path=LOG_CSV,
            plot_dir="./mt5-pretrain/plots",
        ),
        # Stop training if eval_loss does not improve for 3 consecutive evaluations.
        # Each evaluation is every eval_steps=10,000 steps, so patience=3 means
        # training stops after 30,000 steps of no improvement.
        # Requires load_best_model_at_end=True and metric_for_best_model="eval_loss".
       # EarlyStoppingCallback(
           # early_stopping_patience=3,
         #   early_stopping_threshold=0.001,  # minimum improvement in eval_loss to count as progress
        #),
    ],
)

# resume_from_checkpoint automatically picks up from the last saved
# checkpoint if training was interrupted (power outage, OOM, etc.).
# On a fresh run with no checkpoints, it starts from scratch.
last_checkpoint = None
if os.path.isdir("./mt5-pretrain"):
    checkpoints = [d for d in os.listdir("./mt5-pretrain") if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join("./mt5-pretrain", sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)


# ---- Per-language evaluation ----
# Evaluate each language/domain separately to see which improved most.
# This gives a breakdown table for the paper.

print("\n" + "="*60)
print("  Per-Language Evaluation (best checkpoint)")
print("="*60)

per_lang_results = []
dataset_names = list(DATASET_DIRECTORY.keys())

for i, name in enumerate(dataset_names):
    results = trainer.evaluate(eval_dataset=val_datasets[i])
    loss = results["eval_loss"]
    ppl = math.exp(min(loss, 20))
    per_lang_results.append({"dataset": name, "eval_loss": loss, "perplexity": ppl})
    print(f"  {name:20s}  eval_loss={loss:.4f}  perplexity={ppl:.2f}")

# Save per-language results to CSV
per_lang_csv = "./mt5-pretrain/per_language_eval.csv"
with open(per_lang_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "eval_loss", "perplexity"])
    for r in per_lang_results:
        writer.writerow([r["dataset"], f"{r['eval_loss']:.6f}", f"{r['perplexity']:.4f}"])
print(f"\nPer-language results saved → {per_lang_csv}")

# Save the final model
trainer.save_model("./mt5-pretrain/final_model")
tokenizer.save_pretrained("./mt5-pretrain/final_model")
print("Final model saved → ./mt5-pretrain/final_model")