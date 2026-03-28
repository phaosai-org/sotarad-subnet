"""
Mock Chutes Inference Server – SotaRad development server
==========================================================

Hosts the base radiology vision-language model locally and exposes an
OpenAI-compatible chat-completions endpoint so validator.py can run inference
without a real Chutes deployment.

Model
-----
Default: mervinpraison/Llama-3.2-11B-Vision-Radiology-mini
  (mllama / Llama-3.2-11B Vision, fine-tuned on radiology data)

Endpoint
--------
  POST /v1/chat/completions
      Body:  OpenAI chat-completions payload with vision content block
      Returns: {"choices": [{"message": {"content": "positive" | "negative"}}]}

  The server ignores the "model" field in the request body – it always routes
  to the single model loaded at startup.

Usage
-----
  # GPU (recommended – model is BF16/4-bit):
  python mock/chutes.py

  # CPU-only (slow but functional):
  python mock/chutes.py --cpu

  # Custom model or port:
  python mock/chutes.py --model user/repo --port 8200

  Then run the validator with:
    python validator.py --mock ...
"""

import argparse
import io
import time
from contextlib import asynccontextmanager
from typing import Any

import requests as req_lib
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


# ── Global model state ────────────────────────────────────────────────────────

_model: MllamaForConditionalGeneration | None = None
_processor: AutoProcessor | None = None
_device: str = "cpu"


def _load_model(model_id: str, cpu_only: bool) -> None:
    global _model, _processor, _device

    print(f"[mock/chutes] Loading model: {model_id}")
    print(f"[mock/chutes] CPU-only: {cpu_only}")

    _processor = AutoProcessor.from_pretrained(model_id)

    dtype = torch.float32 if cpu_only else torch.bfloat16

    if cpu_only or not torch.cuda.is_available():
        _device = "cpu"
        print("[mock/chutes] Running on CPU (this will be slow for an 11B model)")
        _model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cpu",
        )
    else:
        _device = "cuda"
        print(f"[mock/chutes] Running on GPU: {torch.cuda.get_device_name(0)}")
        _model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )

    _model.eval()
    print(f"[mock/chutes] Model ready on {_device}")


# ── Inference ─────────────────────────────────────────────────────────────────

def _fetch_image(image_url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image."""
    resp = req_lib.get(image_url, timeout=15, stream=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _run_inference(image_url: str, prompt_text: str, max_new_tokens: int = 50) -> str:
    """Run vision-language inference and return the generated text."""
    image = _fetch_image(image_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    input_text = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(image, input_text, return_tensors="pt").to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _processor.decode(new_tokens, skip_special_tokens=True).strip()


# ── FastAPI app ───────────────────────────────────────────────────────────────

def _build_app() -> FastAPI:
    app = FastAPI(title="SotaRad Mock Chutes Inference Server", version="1.0-mock")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": _model is not None,
            "device": _device,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any]):
        if _model is None or _processor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # ── Extract image URL and text from the request ───────────────────────
        messages: list[dict] = body.get("messages", [])
        image_url: str | None = None
        prompt_text: str = ""

        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "image_url":
                        image_url = block["image_url"]["url"]
                    elif block.get("type") == "text":
                        prompt_text = block.get("text", "")
            elif isinstance(content, str):
                prompt_text = content

        if not image_url:
            raise HTTPException(status_code=400, detail="No image_url found in request")

        # ── Run inference ─────────────────────────────────────────────────────
        max_tokens = int(body.get("max_tokens", 50))
        t0 = time.time()
        try:
            generated = _run_inference(image_url, prompt_text, max_new_tokens=max_tokens)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

        latency_ms = int((time.time() - t0) * 1000)
        print(
            f"[mock/chutes] inference  image={image_url.split('/')[-1]}  "
            f"response={generated!r}  latency={latency_ms}ms"
        )

        # ── Return OpenAI-compatible response ─────────────────────────────────
        return JSONResponse({
            "id":      "mock-cmpl-000",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   "mock-radiology",
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": generated},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": max_tokens, "total_tokens": max_tokens},
        })

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SotaRad mock Chutes inference server")
    parser.add_argument(
        "--model",
        default="mervinpraison/Llama-3.2-11B-Vision-Radiology-mini",
        help="Hugging Face model ID to load (default: mervinpraison/Llama-3.2-11B-Vision-Radiology-mini)",
    )
    parser.add_argument(
        "--port", type=int, default=8200,
        help="Port to listen on (default: 8200)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only inference (slow but works without a GPU)",
    )
    args = parser.parse_args()

    _load_model(args.model, cpu_only=args.cpu)

    print(f"[mock/chutes] Starting on port {args.port}")
    app = _build_app()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
