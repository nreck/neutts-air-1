import warnings
import os
import platform
import re
from functools import partial

import torch
import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from fire import Fire
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from loguru import logger as LOGGER
from datasets import load_from_disk


warnings.filterwarnings("ignore")


ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


def data_filter(sample):
    text = sample["text"]

    if len(text) == 0:
        return False

    if re.search(r"\d", text):
        return False

    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False

    if text[-1] not in ".,?!":
        return False

    if "£" in text or "$" in text:
        return False

    return True


def _maybe_set_espeak_library():
    # On macOS, espeak may be installed but not linked or not found by phonemizer.
    if platform.system().lower() == "darwin":
        lib_path = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        if lib_path and os.path.exists(lib_path):
            try:
                EspeakWrapper.set_library(lib_path)
                LOGGER.info(f"Using espeak library from env: {lib_path}")
                return
            except Exception as e:
                LOGGER.warning(f"Failed to set espeak library from env: {e}")
        # Fallback to common Homebrew path for espeak
        default_lib = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib"
        if os.path.exists(default_lib):
            try:
                EspeakWrapper.set_library(default_lib)
                LOGGER.info(f"Using espeak library: {default_lib}")
            except Exception as e:
                LOGGER.warning(f"Failed to set default espeak library: {e}")


def preprocess_sample(sample, tokenizer, max_len, g2p):
    # get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    ignore_index = -100  # this is from LLaMA

    # unpack sample
    vq_codes = sample["codes"]
    text = sample["text"]

    # phonemize (Danish)
    phones = g2p.phonemize([text])
    if not phones or not phones[0]:
        LOGGER.warning(f"Empty phonemization for sample: {sample.get('__key__', '')}")
        return None

    phones = phones[0].split()
    phones = " ".join(phones)

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    # prepend language token <|DA|> for Danish
    chat = (
        f"user: Convert the text to speech: <|DA|><|TEXT_PROMPT_START|>{phones}"
        f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        f"<|SPEECH_GENERATION_END|>"
    )
    ids = tokenizer.encode(chat)

    # pad to make seq len
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    # convert to tensor
    input_ids = torch.tensor(ids, dtype=torch.long)

    labels = torch.full_like(input_ids, ignore_index)
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

    # create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str, dataset_path: str = "datasets/coral-da-neucodec-5k"):
    _maybe_set_espeak_library()

    LOGGER.info(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    LOGGER.info(f"Saving to: {checkpoints_dir}")

    restore_from = config.restore_from
    LOGGER.info(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    # Ensure language/control token for Danish exists
    new_tokens = ["<|DA|>"]
    n_added = tokenizer.add_tokens(new_tokens)
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.vocab_size = len(tokenizer)
        LOGGER.info(f"Added {n_added} new tokens: {new_tokens}")

    # Danish espeak backend
    g2p = phonemizer.backend.EspeakBackend(
        language="da",
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
    )

    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )

    LOGGER.info(f"Loading local dataset from {dataset_path}")
    ds = load_from_disk(dataset_path)
    ds = ds.filter(lambda s: data_filter({"text": s["text"]})).map(
        partial_preprocess, remove_columns=["text", "codes"]
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=True,
        dataloader_num_workers=8,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(checkpoints_dir)


if __name__ == "__main__":
    Fire(main)


