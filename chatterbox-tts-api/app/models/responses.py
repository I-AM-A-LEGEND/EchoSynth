"""
Response models for API validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str
    model_loaded: bool
    device: str
    config: Dict[str, Any]
    memory_info: Optional[Dict[str, float]] = None
    initialization_state: Optional[str] = None
    initialization_progress: Optional[str] = None
    initialization_error: Optional[str] = None


class ModelInfo(BaseModel):
    """Individual model information"""
    
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Models listing response"""
    
    object: str
    data: List[ModelInfo]


class ConfigResponse(BaseModel):
    """Configuration response model"""
    
    api_info: Dict[str, Any]
    server: Dict[str, Any]
    model: Dict[str, Any]
    defaults: Dict[str, Any]
    memory_management: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: Dict[str, str]


# SSE Response Models for OpenAI compatibility
class SSEUsageInfo(BaseModel):
    """Usage information for SSE completion event"""
    
    input_tokens: int
    output_tokens: int  
    total_tokens: int


class SSEAudioInfo(BaseModel):
    """SSE audio metadata event model"""
    
    type: str = "speech.audio.info"
    sample_rate: int
    channels: int
    bits_per_sample: int


class SSEAudioDelta(BaseModel):
    """SSE audio delta event model"""
    
    type: str = "speech.audio.delta"
    audio: str  # Base64 encoded audio chunk


class SSEAudioDone(BaseModel):
    """SSE audio completion event model"""
    
    type: str = "speech.audio.done"
    usage: SSEUsageInfo


class TTSProgressResponse(BaseModel):
    """TTS progress response model"""
    
    current_chunk: int
    total_chunks: int
    current_step: str
    progress_percentage: float
    estimated_completion: Optional[float] = None


class TTSStatusResponse(BaseModel):
    """TTS status response model"""
    
    status: str
    is_processing: bool
    request_id: Optional[str] = None
    start_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    text_length: Optional[int] = None
    text_preview: Optional[str] = None
    voice_source: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    progress: Optional[TTSProgressResponse] = None
    error_message: Optional[str] = None
    memory_usage: Optional[Dict[str, float]] = None
    total_requests: int = 0
    message: Optional[str] = None


class TTSStatisticsResponse(BaseModel):
    """TTS statistics response model"""
    
    total_requests: int
    completed_requests: int
    error_requests: int
    success_rate: float
    average_duration_seconds: float
    average_text_length: float
    is_processing: bool


class APIInfoResponse(BaseModel):
    """API information response model"""

    api_name: str
    version: str
    status: str
    tts_status: TTSStatusResponse
    statistics: TTSStatisticsResponse
    memory_info: Optional[Dict[str, float]] = None
    recent_requests: Optional[List[Dict[str, Any]]] = None
    uptime_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class STTResponse(BaseModel):
    """Speech-to-text response model"""

    text: str = Field(description="The transcribed text from the audio")
    language: Optional[str] = Field(None, description="Detected or specified language code")
    confidence: Optional[float] = Field(None, description="Transcription confidence score (0.0 to 1.0)")


class STTStatusResponse(BaseModel):
    """STT status response model"""

    state: str
    ready: bool
    progress: str
    error: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None
