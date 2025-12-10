<<<<<<< HEAD
"""
FastAPI Victim Service with Timing Obfuscation Support

Environment Variables:
  MODEL_NAME          - Model to serve (fake_a, fake_b, distilgpt2, etc.)
  USE_REAL_MODELS     - 0 or 1
  COVERT_ENABLED      - 0 or 1
  OBFUSCATION_STRATEGY - none, random, bucket, constant
  OBFUSCATION_PARAM   - Strategy parameter (ms)
"""

import os
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
=======
# victim_service/server.py

import csv
import os
import time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b

from model_backend import ModelBackend
from covert_channel import maybe_send_covert

<<<<<<< HEAD
# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.environ.get("MODEL_NAME", "fake_a")
USE_REAL_MODELS = os.environ.get("USE_REAL_MODELS", "0") == "1"
COVERT_ENABLED = os.environ.get("COVERT_ENABLED", "0") == "1"
COVERT_SLOT_MS = float(os.environ.get("COVERT_SLOT_MS", "2.0"))

# Timing obfuscation settings
OBFUSCATION_STRATEGY = os.environ.get("OBFUSCATION_STRATEGY", "none")
OBFUSCATION_PARAM = float(os.environ.get("OBFUSCATION_PARAM", "0"))

LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_REQUESTS = os.environ.get("LOG_REQUESTS", "1") == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: Optional[int] = Field(20, description="Max tokens to generate")
    steps: Optional[int] = Field(None, description="Computation steps (fake models)")
    tag: Optional[str] = Field(None, description="Request tag for tracking")
=======
app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "fake_a")
COVERT_ENABLED = os.getenv("COVERT_ENABLED", "0") == "1"
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQ_LOG_PATH = LOG_DIR / "requests.csv"

backend = ModelBackend(MODEL_NAME)


class GenerateRequest(BaseModel):
    prompt: str
    steps: int | None = None  # for fake models controls CPU work
    tag: str | None = None    # optional experiment label
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b


class GenerateResponse(BaseModel):
    model_name: str
    output: str
    elapsed_ms: float
<<<<<<< HEAD
    actual_elapsed_ms: float
    obfuscation_strategy: str
    covert_triggered: bool
    covert_topic: Optional[str] = None
    tag: Optional[str] = None
    is_real_model: bool


class HealthResponse(BaseModel):
    status: str
    model_name: str
    is_real_model: bool
    covert_enabled: bool
    obfuscation_strategy: str
    obfuscation_param: float
    timestamp: str


# =============================================================================
# Global State
# =============================================================================

backend: Optional[ModelBackend] = None
request_log_path: Optional[Path] = None
request_count: int = 0


# =============================================================================
# Logging
# =============================================================================

def _ensure_log_dir():
    global request_log_path
    if not LOG_REQUESTS:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    request_log_path = LOG_DIR / f"requests_{MODEL_NAME}_{timestamp}.csv"
    with open(request_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "request_id", "model_name", "tag", "prompt_length", 
                        "actual_elapsed_ms", "obfuscated_elapsed_ms", "covert_triggered"])
    logger.info(f"Request logging to: {request_log_path}")


def _append_request_log(request_id, tag, prompt_length, actual_ms, obfuscated_ms, covert_triggered):
    if not LOG_REQUESTS or request_log_path is None:
        return
    try:
        with open(request_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), request_id, MODEL_NAME, tag or "",
                           prompt_length, f"{actual_ms:.3f}", f"{obfuscated_ms:.3f}", covert_triggered])
    except Exception as e:
        logger.warning(f"Failed to write request log: {e}")


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend
    
    logger.info(f"Starting victim service: model={MODEL_NAME}, real={USE_REAL_MODELS}")
    logger.info(f"Obfuscation: strategy={OBFUSCATION_STRATEGY}, param={OBFUSCATION_PARAM}")
    
    backend = ModelBackend(
        MODEL_NAME, 
        use_real=USE_REAL_MODELS,
        obfuscation_strategy=OBFUSCATION_STRATEGY,
        obfuscation_param=OBFUSCATION_PARAM
    )
    
    _ensure_log_dir()
    
    if USE_REAL_MODELS:
        logger.info("Pre-warming model...")
        try:
            backend.generate("warmup", max_new_tokens=5)
            logger.info("Model pre-warm complete")
        except Exception as e:
            logger.warning(f"Pre-warm failed: {e}")
    
    yield
    logger.info("Shutting down victim service")


app = FastAPI(title="LLM Victim Service", version="3.0.0", lifespan=lifespan)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_name=MODEL_NAME,
        is_real_model=USE_REAL_MODELS,
        covert_enabled=COVERT_ENABLED,
        obfuscation_strategy=OBFUSCATION_STRATEGY,
        obfuscation_param=OBFUSCATION_PARAM,
        timestamp=datetime.now().isoformat()
    )


@app.get("/models")
async def list_models():
    return ModelBackend.list_models()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    global request_count
    
    if backend is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    request_count += 1
    
    # Covert channel
    covert_triggered = False
    covert_topic = None
    if COVERT_ENABLED:
        covert_triggered, covert_topic, _ = maybe_send_covert(
            request.prompt, enabled=True, slot_ms=COVERT_SLOT_MS)
    
    # Generate with timing obfuscation
    if USE_REAL_MODELS:
        output, actual_ms, obfuscated_ms = backend.generate(
            request.prompt, max_new_tokens=request.max_new_tokens or 20)
    else:
        output, actual_ms, obfuscated_ms = backend.generate(
            request.prompt, steps=request.steps)
    
    _append_request_log(request_count, request.tag, len(request.prompt),
                       actual_ms, obfuscated_ms, covert_triggered)
    
    return GenerateResponse(
        model_name=MODEL_NAME,
        output=output,
        elapsed_ms=obfuscated_ms,
        actual_elapsed_ms=actual_ms,
        obfuscation_strategy=OBFUSCATION_STRATEGY,
        covert_triggered=covert_triggered,
        covert_topic=covert_topic,
        tag=request.tag,
        is_real_model=USE_REAL_MODELS
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
=======
    covert_triggered: bool
    tag: str | None


def _append_request_log(
    prompt: str,
    tag: str | None,
    elapsed_ms: float,
    covert_triggered: bool,
) -> None:
    """Append one line to a CSV log (request-level metadata)."""
    with REQ_LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                time.time(),  # unix timestamp
                MODEL_NAME,
                tag or "",
                len(prompt),
                f"{elapsed_ms:.3f}",
                int(covert_triggered),
            ]
        )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    prompt = req.prompt
    steps = req.steps
    tag = req.tag

    covert_triggered = maybe_send_covert(prompt, enabled=COVERT_ENABLED)

    start = time.time()
    output = backend.generate(prompt, steps=steps)
    elapsed_ms = (time.time() - start) * 1000.0

    _append_request_log(prompt, tag, elapsed_ms, covert_triggered)

    return GenerateResponse(
        model_name=MODEL_NAME,
        output=output,
        elapsed_ms=elapsed_ms,
        covert_triggered=covert_triggered,
        tag=tag,
    )
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b
