"""
Main FastAPI application
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.tts_model import initialize_model
from app.core.stt_model import initialize_stt_model
from app.core.voice_library import get_voice_library
from app.api.router import api_router
from app.config import Config


ascii_art = r"""
  ____ _           _   _            _               
 / ___| |__   __ _| |_| |_ ___ _ __| |__   _____  __
| |   | '_ \ / _` | __| __/ _ \ '__| '_ \ / _ \ \/ /
| |___| | | | (_| | |_| ||  __/ |  | |_) | (_) >  < 
 \____|_| |_|\__,_|\__|\__\___|_|  |_.__/ \___/_/\_\
                                                    
"""


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(ascii_art)
    
    # Start model initialization in the background
    # This allows the server to respond to health checks immediately
    # while the model loads asynchronously
    import asyncio
    tts_init_task = asyncio.create_task(initialize_model())
    stt_init_task = asyncio.create_task(initialize_stt_model())
    
    # Initialize voice library to restore default voice settings
    print("Initializing voice library...")
    voice_lib = get_voice_library()
    default_voice = voice_lib.get_default_voice()
    if default_voice:
        print(f"Restored default voice: {default_voice}")
    else:
        print("Using system default voice")
    
    # Note: We don't await the model initialization here
    # The server will start immediately and health checks will show initialization status
    
    yield
    
    # Shutdown (cleanup if needed)
    # Cancel model initialization if they're still running
    if not tts_init_task.done():
        tts_init_task.cancel()
        try:
            await tts_init_task
        except asyncio.CancelledError:
            pass

    if not stt_init_task.done():
        stt_init_task.cancel()
        try:
            await stt_init_task
        except asyncio.CancelledError:
            pass


# Create FastAPI app
app = FastAPI(
    title="Chatterbox TTS API",
    description="REST API for Chatterbox TTS with OpenAI-compatible endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
cors_origins = Config.CORS_ORIGINS
if cors_origins == "*":
    allowed_origins = ["*"]
else:
    # Split comma-separated origins and strip whitespace
    allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(api_router)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error"
            }
        }
    )
