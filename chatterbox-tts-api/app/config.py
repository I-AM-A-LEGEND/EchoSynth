"""
Configuration management for Chatterbox TTS API
"""

import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class"""
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 4123))
    
    # TTS Model settings
    EXAGGERATION = float(os.getenv('EXAGGERATION', 0.5))
    CFG_WEIGHT = float(os.getenv('CFG_WEIGHT', 0.5))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
    
    # Text processing
    MAX_CHUNK_LENGTH = int(os.getenv('MAX_CHUNK_LENGTH', 280))
    MAX_TOTAL_LENGTH = int(os.getenv('MAX_TOTAL_LENGTH', 3000))
    
    # Voice and model settings
    VOICE_SAMPLE_PATH = os.getenv('VOICE_SAMPLE_PATH', 'chatterbox-tts-api/assets/female_american.flac')
    DEVICE_OVERRIDE = os.getenv('DEVICE', 'auto')
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')

    # STT model settings
    STT_MODEL_NAME = os.getenv('STT_MODEL_NAME', 'openai/whisper-tiny')
    STT_MODEL_CACHE_DIR = os.getenv('STT_MODEL_CACHE_DIR', MODEL_CACHE_DIR)
    STT_DEVICE_OVERRIDE = os.getenv('STT_DEVICE', DEVICE_OVERRIDE)

    # Voice library settings
    VOICE_LIBRARY_DIR = os.getenv('VOICE_LIBRARY_DIR', './voices')
    
    # Memory management settings
    MEMORY_CLEANUP_INTERVAL = int(os.getenv('MEMORY_CLEANUP_INTERVAL', 5))
    CUDA_CACHE_CLEAR_INTERVAL = int(os.getenv('CUDA_CACHE_CLEAR_INTERVAL', 3))
    ENABLE_MEMORY_MONITORING = os.getenv('ENABLE_MEMORY_MONITORING', 'true').lower() == 'true'
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    @classmethod
    def validate(cls):
        """Validate configuration values"""
        if not (0.25 <= cls.EXAGGERATION <= 2.0):
            raise ValueError(f"EXAGGERATION must be between 0.25 and 2.0, got {cls.EXAGGERATION}")
        if not (0.0 <= cls.CFG_WEIGHT <= 1.0):
            raise ValueError(f"CFG_WEIGHT must be between 0.0 and 1.0, got {cls.CFG_WEIGHT}")
        if not (0.05 <= cls.TEMPERATURE <= 5.0):
            raise ValueError(f"TEMPERATURE must be between 0.05 and 5.0, got {cls.TEMPERATURE}")
        if cls.MAX_CHUNK_LENGTH <= 0:
            raise ValueError(f"MAX_CHUNK_LENGTH must be positive, got {cls.MAX_CHUNK_LENGTH}")
        if cls.MAX_TOTAL_LENGTH <= 0:
            raise ValueError(f"MAX_TOTAL_LENGTH must be positive, got {cls.MAX_TOTAL_LENGTH}")
        if cls.MEMORY_CLEANUP_INTERVAL <= 0:
            raise ValueError(f"MEMORY_CLEANUP_INTERVAL must be positive, got {cls.MEMORY_CLEANUP_INTERVAL}")
        if cls.CUDA_CACHE_CLEAR_INTERVAL <= 0:
            raise ValueError(f"CUDA_CACHE_CLEAR_INTERVAL must be positive, got {cls.CUDA_CACHE_CLEAR_INTERVAL}")


def detect_device():
    """Detect the best available device"""
    if Config.DEVICE_OVERRIDE.lower() != 'auto':
        return Config.DEVICE_OVERRIDE.lower()
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
