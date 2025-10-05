"""
STT model initialization and management
"""

import asyncio
from enum import Enum
from typing import Optional
from transformers import pipeline
from app.config import Config, detect_device

# Global STT model instance
_stt_pipeline = None
_stt_device = None
_stt_initialization_state = "not_started"
_stt_initialization_error = None
_stt_initialization_progress = ""


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


def _detect_stt_device():
    """Detect the best available device for STT"""
    print(f"STT_DEVICE_OVERRIDE from config: {Config.STT_DEVICE_OVERRIDE}")

    if Config.STT_DEVICE_OVERRIDE and Config.STT_DEVICE_OVERRIDE.lower() != 'auto':
        print(f"Using STT device override: {Config.STT_DEVICE_OVERRIDE}")
        if Config.STT_DEVICE_OVERRIDE.lower() == 'cpu':
            return -1
        elif Config.STT_DEVICE_OVERRIDE.lower() == 'cuda':
            return 0
        else:
            return Config.STT_DEVICE_OVERRIDE.lower()

    # Prefer GPU for speed when auto-detecting
    base_device = detect_device()
    print(f"Base device detected: {base_device}")
    if base_device == 'cuda':
        return 0  # GPU device index for transformers
    else:
        return -1  # CPU for transformers


async def initialize_stt_model():
    """Initialize the STT model"""
    global _stt_pipeline, _stt_device, _stt_initialization_state, _stt_initialization_error, _stt_initialization_progress

    try:
        _stt_initialization_state = InitializationState.INITIALIZING.value
        _stt_initialization_progress = "Initializing STT model..."

        _stt_device = _detect_stt_device()

        print(f"Initializing STT model...")
        print(f"STT Model: {Config.STT_MODEL_NAME}")
        print(f"STT Device: {_stt_device}")

        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        _stt_pipeline = await loop.run_in_executor(
            None,
            lambda: pipeline(
                "automatic-speech-recognition",
                model=Config.STT_MODEL_NAME,
                torch_dtype="auto",
                device=_stt_device
            )
        )

        _stt_initialization_state = InitializationState.READY.value
        _stt_initialization_progress = "STT model ready"
        _stt_initialization_error = None
        print("✓ STT model initialized successfully")
    except Exception as e:
        _stt_initialization_state = InitializationState.ERROR.value
        _stt_initialization_error = str(e)
        _stt_initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize STT model: {e}")
        raise e


def get_stt_pipeline():
    """Get the current STT pipeline instance"""
    return _stt_pipeline


def get_stt_device():
    """Get the current STT device"""
    return _stt_device


def get_stt_initialization_state():
    """Get the current STT initialization state"""
    return _stt_initialization_state


def get_stt_initialization_progress():
    """Get the current STT initialization progress message"""
    return _stt_initialization_progress


def get_stt_initialization_error():
    """Get the STT initialization error if any"""
    return _stt_initialization_error


def is_stt_ready():
    """Check if the STT model is ready for use"""
    return _stt_initialization_state == InitializationState.READY.value and _stt_pipeline is not None


def is_stt_initializing():
    """Check if the STT model is currently initializing"""
    return _stt_initialization_state == InitializationState.INITIALIZING.value
