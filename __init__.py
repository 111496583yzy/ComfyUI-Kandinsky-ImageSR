"""ComfyUI adapter for the official Kandinsky ImageSR 2.0 Diffusers pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from comfy import model_management
import folder_paths


HERE = Path(__file__).parent


def _settings():
    with (HERE / "deployment.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _model_directory(config):
    directory = Path(config["model_directory"]).expanduser()
    if not directory.is_absolute():
        directory = Path(folder_paths.models_dir) / directory
    return directory


class KandinskyImageSR2:
    @classmethod
    def INPUT_TYPES(cls):
        config = _settings()
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": ([f"{factor}x" for factor in config["upscale_factors"]],),
                "creativity": ("INT", {
                    "default": config["default_creativity"],
                    "min": 0,
                    "max": config["max_creativity"],
                    "tooltip": "Lower values stay closer to the source; higher values let the model invent more detail.",
                }),
            },
            "optional": {
                "overlap": ("FLOAT", {
                    "default": config["default_overlap"],
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Overlap between neighboring tiles, e.g. 0.25 = 25%. Larger values can help with seams but take longer.",
                }),
                "tile_batch_size": ("INT", {
                    "default": config["default_tile_batch_size"],
                    "min": 1,
                    "step": 1,
                    "tooltip": "Number of tiles processed together. Larger batches use more GPU memory.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Prompt-free super-resolution using the pinned official Kandinsky ImageSR 2.0 pipeline."

    def upscale(self, image, upscale_factor, creativity, overlap=None, tile_batch_size=None):
        config = _settings()
        factors = {f"{factor}x": factor for factor in config["upscale_factors"]}
        if upscale_factor not in factors or not 0 <= creativity <= config["max_creativity"]:
            raise ValueError("Choose a scale and creativity level listed for the installed Kandinsky model")
        if overlap is not None:
            overlap = float(overlap)
            if not 0.0 <= overlap < 1.0:
                raise ValueError("Tile overlap must be at least 0 and below 1")
        if tile_batch_size is not None and (type(tile_batch_size) is not int or tile_batch_size < 1):
            raise ValueError("Tile batch size must be a positive integer")
        python = Path(sys.executable)
        model = _model_directory(config)
        if not python.is_file() or not (model / "model_index.json").is_file():
            raise RuntimeError("The ComfyUI Python executable or the installed Kandinsky model is missing")
        if image.ndim != 4 or not image.numel() or image.shape[-1] != 3:
            raise ValueError("Kandinsky ImageSR expects an RGB image or RGB image batch")

        device = model_management.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("Kandinsky ImageSR on this installation needs a CUDA device")

        # Drop Comfy's idle model cache before giving the device to the standalone
        # runner. The node runs in ComfyUI's normal queue; other workflows can
        # reload their cached weights after the runner exits.
        model_management.unload_all_models()
        model_management.soft_empty_cache(force=True)
        model_management.throw_exception_if_processing_interrupted()

        factor = factors[upscale_factor]
        timeout = float(config["seconds_per_image"]) * image.shape[0]
        scratch = Path(folder_paths.get_temp_directory())
        scratch.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="kandinsky-isr-", dir=scratch) as directory:
            temp = Path(directory)
            source, target, log = (temp / name for name in ("input.npy", "output.npy", "infer.log"))
            np.save(source, image.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy(), allow_pickle=False)
            command = [
                str(python), "-I", str(HERE / "worker.py"),
                "--model", str(model), "--input", str(source), "--output", str(target),
                "--device", str(device), "--factor", str(factor), "--creativity", str(creativity),
            ]
            if overlap is not None:
                command.extend(["--overlap", str(overlap)])
            if tile_batch_size is not None:
                command.extend(["--tile-batch-size", str(tile_batch_size)])
            deadline = time.monotonic() + timeout
            with log.open("wb") as stream, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as runner:
                try:
                    while runner.poll() is None:
                        model_management.throw_exception_if_processing_interrupted()
                        if time.monotonic() > deadline:
                            raise RuntimeError("Kandinsky ImageSR exceeded its configured run time")
                        try:
                            runner.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if runner.poll() is None:
                        runner.terminate()
                        try:
                            runner.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            runner.kill()
                            runner.wait()
            if runner.returncode != 0 or not target.is_file():
                detail = log.read_text(encoding="utf-8", errors="replace").strip()
                raise RuntimeError("Kandinsky ImageSR runner failed:\n" + (detail[-4500:] or f"exit {runner.returncode}"))
            result = np.load(target, allow_pickle=False)
            batch, height, width, channels = image.shape
            if result.dtype != np.float32 or result.shape != (batch, height * factor, width * factor, channels):
                raise RuntimeError(f"Kandinsky ImageSR returned an unexpected image array: {result.shape}")
            if not np.isfinite(result).all():
                raise RuntimeError("Kandinsky ImageSR returned invalid pixels")
            logging.getLogger(__name__).info(
                "Kandinsky ImageSR tiling: scale=%sx overlap=%s tile_batch_size=%s",
                factor, config["default_overlap"] if overlap is None else overlap,
                config["default_tile_batch_size"] if tile_batch_size is None else tile_batch_size,
            )
            return (torch.from_numpy(result),)


NODE_CLASS_MAPPINGS = {"KandinskyImageSR2": KandinskyImageSR2}
NODE_DISPLAY_NAME_MAPPINGS = {"KandinskyImageSR2": "Kandinsky ImageSR 2.0 (图片超分)"}
