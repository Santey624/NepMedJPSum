import math, glob, os
import numpy as np
import torch
from itertools import chain
from dataclasses import dataclass
from typing import Dict, List
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer,
    TrainingArguments, PreTrainedTokenizerBase,
)
from datasets import load_dataset

# ---- Helpers (copied from pretraining_script.py) ----

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

# ---- Config ----
CHECKPOINT = "./mt5-pretrain/final_model"
BASE_MODEL = "google/mt5-xl"
MAX_SEQ_LENGTH = 1024
EVAL_SAMPLES = 5000

DATASET_DIRECTORY = {
    "eng_medical_data": "/workspace/Nepmedjp/pre-training/pre-trainingdataset/EnglishMedicalDataset",
    "ne_medical_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/Nepalipubmeddata",
    "jp_medical_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/Japanesemedicaldatafinal",
    "ne_general_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/generalnepalidata",
    "jp_general_data":  "/workspace/Nepmedjp/pre-training/pre-trainingdataset/GeneralJapanese",
}

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
expanded_inputs_length, targets_length = compute_input_and_target_lengths(MAX_SEQ_LENGTH)

data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=MAX_SEQ_LENGTH,
    target_length=targets_length,
    pad_token_id=tokenizer.pad_token_id,
    decoder_start_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
)

def prepare_eval_dataset(path, n_samples=EVAL_SAMPLES):
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    ds = load_dataset("json", data_files=files, split="train")
    if "text" not in ds.column_names and "test" in ds.column_names:
        ds = ds.rename_column("test", "text")
    cols = [c for c in ds.column_names if c != "text"]
    if cols:
        ds = ds.remove_columns(cols)
    split = ds.train_test_split(test_size=0.05, seed=42)
    val = split["test"]
    if len(val) > n_samples:
        val = val.shuffle(seed=42).select(range(n_samples))
    val = val.map(
        lambda ex: tokenizer(ex["text"], return_attention_mask=False),
        batched=True, remove_columns=["text"], num_proc=2,
    )
    val = val.map(
        lambda ex: group_texts(ex, expanded_inputs_length),
        batched=True, num_proc=2,
    )
    return val

# ---- Prepare eval sets ----
eval_sets = {}
for name, path in DATASET_DIRECTORY.items():
    eval_sets[name] = prepare_eval_dataset(path)
    print(f"{name}: {len(eval_sets[name])} eval chunks")

# ---- Evaluate base vs checkpoint ----
for model_label, model_path in [("base_mt5xl", BASE_MODEL), ("checkpoint-10k", CHECKPOINT)]:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_label}")
    print(f"{'='*60}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/eval_tmp",
            per_device_eval_batch_size=4,
            bf16=True,
            prediction_loss_only=True,
            report_to="none",
        ),
        data_collator=data_collator,
    )
    for name, val_ds in eval_sets.items():
        results = trainer.evaluate(eval_dataset=val_ds)
        loss = results["eval_loss"]
        ppl = math.exp(min(loss, 20))
        print(f"  {name:20s}  loss={loss:.4f}  ppl={ppl:.2f}")
    del model
    torch.cuda.empty_cache()