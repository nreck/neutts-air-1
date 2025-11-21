import os
import math
from typing import Optional, Iterator, Dict, Any, List

import torch
import numpy as np
from datasets import load_dataset, Dataset, Features, Value, Sequence
from loguru import logger as LOGGER
from fire import Fire
import librosa
import io
import soundfile as sf

from neucodec import NeuCodec


# Copied from examples/finetune.py to ensure identical filtering behavior
import re

ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


def data_filter(text: str) -> bool:
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


def _ensure_mono_16k(audio_array: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if audio_array.ndim > 1:
        # average channels to mono
        audio_array = np.mean(audio_array, axis=-1)
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
    return audio_array.astype(np.float32, copy=False)


def _iter_coral_stream(split: str, limit: int) -> Iterator[Dict[str, Any]]:
    """
    Stream CoRal TTS from HuggingFace, download parquet files, and manually decode audio bytes.
    This avoids torchcodec by reading raw parquet with pyarrow and decoding audio bytes ourselves.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    
    # Get list of parquet files for this split from HF
    fs = HfFileSystem()
    repo_id = "CoRal-project/coral-tts"
    files = fs.ls(f"datasets/{repo_id}/data", detail=False)
    parquet_files = [f for f in files if f.endswith(".parquet") and f"/{split}-" in f]
    parquet_files = sorted([f.split("/")[-1] for f in parquet_files])
    
    if not parquet_files:
        raise ValueError(f"No parquet files found for split '{split}'")
    
    LOGGER.info(f"Found {len(parquet_files)} parquet files for split '{split}'")
    
    count = 0
    for pq_idx, pq_file in enumerate(parquet_files):
        if limit and count >= limit:
            break
        
        # Download parquet file
        LOGGER.info(f"Downloading parquet file {pq_idx + 1}/{len(parquet_files)}: {pq_file}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"data/{pq_file}",
            repo_type="dataset",
        )
        LOGGER.info(f"Downloaded {pq_file}, reading...")
        
        # Read with pyarrow
        table = pq.read_table(local_path)
        LOGGER.info(f"Read {len(table)} rows from {pq_file}")
        
        for batch in table.to_batches(max_chunksize=100):
            if limit and count >= limit:
                break
            
            batch_dict = batch.to_pydict()
            
            for idx in range(len(batch_dict['text'])):
                if limit and count >= limit:
                    break
                
                text = batch_dict['text'][idx]
                if not data_filter(text):
                    continue
                
                audio_struct = batch_dict['audio'][idx]
                
                # audio_struct is a dict with keys: 'bytes', 'path', 'sampling_rate'
                audio_bytes = audio_struct.get('bytes')
                if not audio_bytes:
                    continue
                
                sampling_rate = audio_struct.get('sampling_rate', 44100)
                
                try:
                    # Decode audio bytes with soundfile
                    audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                    
                    yield {
                        "text": text,
                        "audio_array": audio_array,
                        "sampling_rate": sr,
                        "transcription_id": batch_dict.get('transcription_id', [idx])[idx],
                    }
                    
                    count += 1
                    
                except Exception as e:
                    LOGGER.warning(f"Failed to decode audio: {e}")
                    continue


def _encode_codes_for_samples(
    samples: Iterator[Dict[str, Any]], 
    limit: int, 
    device: str
) -> List[Dict[str, Any]]:
    """
    Encode audio to NeuCodec codes for each sample.
    """
    LOGGER.info("Loading NeuCodec model...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(device)
    LOGGER.info("NeuCodec model loaded, starting encoding...")
    
    data = []
    for idx, sample in enumerate(samples):
        if limit and idx >= limit:
            break
        
        try:
            audio_array = sample["audio_array"]
            sr = sample["sampling_rate"]
            
            # Ensure mono 16kHz
            audio_array = _ensure_mono_16k(audio_array, sr, target_sr=16000)
            
            # Encode to codes
            wav_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0).cpu().numpy()
            
            data.append({
                "text": sample["text"],
                "codes": codes.tolist(),
                "__key__": f"coral_{sample['transcription_id']}",
            })
            
            if (idx + 1) % 10 == 0:
                LOGGER.info(f"Encoded {idx + 1}/{limit} samples")
        
        except Exception as e:
            LOGGER.warning(f"Failed to encode sample {idx}: {e}")
            continue
    
    return data


def main(
    split: str = "train",
    limit: int = 5000,
    out: str = "datasets/coral-da-neucodec-5k",
    device: str = "cpu",
):
    """
    Prepare CoRal TTS dataset by encoding audio to NeuCodec codes.
    Streams from HuggingFace and manually decodes audio bytes to avoid torchcodec.
    
    Args:
        split: Dataset split (default: train)
        limit: Max samples to encode (default: 5000)
        out: Output dataset directory (default: datasets/coral-da-neucodec-5k)
        device: Device for encoding (cpu or cuda, default: cpu)
    """
    LOGGER.info(f"Preparing CoRal TTS split={split}, limit={limit}, device={device}")
    LOGGER.info("Streaming from HuggingFace and manually decoding audio bytes")
    
    # Stream and filter CoRal samples
    samples = _iter_coral_stream(split=split, limit=limit)
    
    # Encode to NeuCodec codes
    data = _encode_codes_for_samples(samples, limit=limit, device=device)
    
    if not data:
        LOGGER.error("No valid samples were encoded. Exiting.")
        return
    
    # Save as HF dataset
    dataset = Dataset.from_list(
        data,
        features=Features({
            "text": Value("string"),
            "codes": Sequence(Value("int64")),
            "__key__": Value("string"),
        })
    )
    
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    dataset.save_to_disk(out)
    LOGGER.info(f"Saved {len(data)} items to {out}")


if __name__ == "__main__":
    Fire(main)
