"""Multimodal capability layer for Sherlock.

Includes:
- Text-to-speech capability wrappers
- Vision/image analysis wrappers
- Image-to-training-record conversion for future fine-tuning
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


try:
    from PIL import Image, ImageStat

    _HAS_PIL = True
except Exception:
    Image = None
    ImageStat = None
    _HAS_PIL = False


@dataclass
class TTSResult:
    ok: bool
    backend: str
    output_path: str | None
    message: str


class SherlockTTS:
    """Text-to-speech with optional local backends.

    Backends are optional and discovered at runtime.
    """

    def speak_to_file(self, text: str, out_path: str = "data/processed/sherlock_tts.txt") -> TTSResult:
        # Stub fallback that still creates a deterministic artifact for pipelines.
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"[TTS_STUB] {text}\n", encoding="utf-8")
        return TTSResult(ok=True, backend="stub", output_path=str(out), message="Stub TTS artifact generated")


@dataclass
class VisionResult:
    ok: bool
    backend: str
    metadata: dict[str, Any]


class SherlockVision:
    """Vision/image analysis wrapper with dependency-free fallback."""

    def analyze_image(self, image_path: str) -> VisionResult:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return VisionResult(ok=False, backend="none", metadata={"error": "file_not_found"})

        payload: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

        if _HAS_PIL:
            try:
                with Image.open(path) as img:
                    payload["width"] = int(img.width)
                    payload["height"] = int(img.height)
                    payload["mode"] = str(img.mode)
                    stat = ImageStat.Stat(img.convert("RGB"))
                    payload["mean_rgb"] = [round(float(v), 4) for v in stat.mean]
                return VisionResult(ok=True, backend="pillow", metadata=payload)
            except Exception as exc:
                payload["warning"] = f"pillow_failed: {type(exc).__name__}"

        return VisionResult(ok=True, backend="binary", metadata=payload)

    def build_image_training_record(self, image_path: str, label: str, notes: str = "") -> dict[str, Any]:
        analysis = self.analyze_image(image_path)
        return {
            "prompt": f"Analyze image for label='{label}'. metadata={analysis.metadata}",
            "completion": f"label={label}; notes={notes}",
            "metadata": {
                "source": "vision",
                "backend": analysis.backend,
                "ok": analysis.ok,
                "image_path": image_path,
            },
        }
