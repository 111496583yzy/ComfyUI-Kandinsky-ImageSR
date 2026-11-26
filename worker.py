"""Load the pinned official pipeline in a subprocess of the same ComfyUI Python runtime."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from kandinsky_image_sr import KandinskyImageSRPipeline


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "input", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--factor", required=True, type=int)
    parser.add_argument("--creativity", required=True, type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--tile-batch-size", type=int)
    options = parser.parse_args()
    tiling = {}
    if options.overlap is not None:
        tiling["overlap"] = options.overlap
    if options.tile_batch_size is not None:
        tiling["tile_batch_size"] = options.tile_batch_size
    device = torch.device(options.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Kandinsky ImageSR needs CUDA")
    source = np.load(options.input, mmap_mode="r", allow_pickle=False)
    batch, height, width, channels = source.shape
    if channels != 3:
        raise ValueError("Kandinsky ImageSR expects three image channels")
    with torch.inference_mode(), torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support the upstream bfloat16 pipeline")
        pipeline = KandinskyImageSRPipeline.from_pretrained(
            str(options.model), dtype=torch.bfloat16, local_files_only=True
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)
        enlarged = (height * options.factor, width * options.factor, channels)
        output = np.lib.format.open_memmap(
            options.output, mode="w+", dtype=np.float32, shape=(batch, *enlarged)
        )
        for index in range(batch):
            frame = np.array(source[index], dtype=np.float32, copy=True)
            output[index] = pipeline(
                frame, upscale_factor=options.factor, creativity=options.creativity, output_type="np", **tiling
            ).images[0]
        output.flush()


if __name__ == "__main__":
    main()
