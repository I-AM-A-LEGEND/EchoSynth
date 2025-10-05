"""
STT endpoint for speech-to-text functionality
"""

import soundfile as sf
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from app.core.stt_model import get_stt_pipeline, is_stt_ready
from app.models.requests import STTRequest
import numpy as np


router = APIRouter()


@router.post("/stt", response_model=None)
async def stt_audio(audio: UploadFile = File(...)):
    """
    Perform speech-to-text on uploaded audio file
    """
    if not is_stt_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT model is not ready. Please try again later."
        )

    if not audio.filename.lower().endswith(('.flac', '.wav', '.mp3', '.m4a', '.ogg', '.webm')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio file format. Supported formats: flac, wav, mp3, m4a, ogg, webm"
        )

    try:
        # Read audio data from uploaded file - Handle different audio formats
        import io
        import tempfile

        # Read all bytes from the uploaded file
        file_contents = await audio.read()

        print(f"Received audio file: {audio.filename}, size: {len(file_contents)} bytes")
        print(f"First 20 bytes: {file_contents[:20].hex()}")

        # Determine audio format based on content first, then extension
        filename = audio.filename.lower()

        # Check for WebM signature (common from browser)
        if len(file_contents) >= 4 and file_contents.startswith(b'\x1A\x45\xDF\xA3'):
            audio_extension = '.webm'
            print("Detected WebM format from file signature")
        elif len(file_contents) >= 12 and file_contents.startswith(b'RIFF') and file_contents[8:12] == b'WAVE':
            audio_extension = '.wav'
            print("Detected WAV format from RIFF header")
        else:
            # Fall back to extension detection
            if filename.endswith('.wav'):
                audio_extension = '.wav'
            elif filename.endswith('.flac'):
                audio_extension = '.flac'
            elif filename.endswith('.m4a') or filename.endswith('.mp4'):
                audio_extension = '.m4a'
            elif filename.endswith('.mp3'):
                audio_extension = '.mp3'
            elif filename.endswith('.ogg') or filename.endswith('.oga'):
                audio_extension = '.ogg'
            else:
                # Default to .wav since that's what the browser claims
                audio_extension = '.wav'
                print("Unknown format, defaulting to .wav extension")

        print(f"Using extension: {audio_extension}")

        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=audio_extension, delete=False) as temp_file:
            temp_file.write(file_contents)
            temp_file_path = temp_file.name

        temp_file_path_debug = temp_file_path
        print(f"Saved temporary file: {temp_file_path}")

        try:
            # Try to read with soundfile first (for supported formats)
            try:
                data, samplerate = sf.read(temp_file_path, dtype='float32')
                print(f"Successfully loaded audio with soundfile: {len(data)} samples at {samplerate}Hz")
            except Exception as sf_error:
                print(f"Soundfile failed: {sf_error}")
                # If soundfile fails, try with pydub as fallback (for webm/opus)
                try:
                    from pydub import AudioSegment
                    import numpy as np

                    # Load with pydub
                    if audio_extension == '.webm':
                        print("Attempting to load WebM with pydub...")
                        audio_segment = AudioSegment.from_file(temp_file_path, format='webm')
                    elif audio_extension == '.wav':
                        print("Attempting to load WAV with pydub...")
                        audio_segment = AudioSegment.from_wav(temp_file_path)
                    elif audio_extension == '.mp3':
                        print("Attempting to load MP3 with pydub...")
                        audio_segment = AudioSegment.from_mp3(temp_file_path)
                    else:
                        raise Exception(f"Unsupported format for pydub: {audio_extension}")

                    print(f"Pydub loaded successfully. Duration: {len(audio_segment)}ms, Channels: {audio_segment.channels}, Rate: {audio_segment.frame_rate}")

                    # Convert to float32 numpy array at 16000 Hz
                    samplerate = 16000
                    data = np.array(audio_segment.set_frame_rate(samplerate).get_array_of_samples(), dtype=np.float32)

                    # Normalize to -1..1 range based on bit depth
                    if audio_segment.channels == 1:
                        bit_depth = audio_segment.sample_width * 8
                        data = data / (2 ** (bit_depth - 1))
                    else:
                        # Handle stereo
                        bit_depth = audio_segment.sample_width * 8
                        data = data.reshape(-1, audio_segment.channels)
                        data = np.mean(data, axis=1)  # Convert to mono
                        data = data / (2 ** (bit_depth - 1))

                    print(f"Successfully processed audio with pydub: {len(data)} samples at {samplerate}Hz")
                except ImportError:
                    raise Exception(f"Soundfile failed and pydub not available. Soundfile error: {sf_error}")
                except Exception as pydub_error:
                    print(f"Pydub also failed: {pydub_error}")
                    # Last resort: try to convert with ffmpeg
                    try:
                        import subprocess
                        import os

                        # Convert to WAV using ffmpeg
                        converted_path = temp_file_path + '_converted.wav'
                        print(f"Attempting ffmpeg conversion: {temp_file_path} -> {converted_path}")
                        subprocess.run([
                            'ffmpeg', '-i', temp_file_path, '-acodec', 'pcm_s16le',
                            '-ar', '16000', '-ac', '1', '-y', converted_path
                        ], check=True, capture_output=True)

                        # Load the converted file
                        data, samplerate = sf.read(converted_path, dtype='float32')
                        print(f"Successfully loaded ffmpeg-converted audio: {len(data)} samples at {samplerate}Hz")

                        # Clean up
                        os.unlink(converted_path)

                    except subprocess.CalledProcessError as ffmpeg_error:
                        print(f"FFmpeg conversion failed: {ffmpeg_error}")
                        raise Exception(f"All audio processing methods failed. Soundfile: {sf_error}, Pydub: {pydub_error}, FFmpeg: {ffmpeg_error}")
                    except Exception as ffmpeg_error:
                        print(f"FFmpeg error: {ffmpeg_error}")
                        raise Exception(f"Both soundfile and pydub failed. Soundfile: {sf_error}, Pydub: {pydub_error}")

        finally:
            # Clean up the temporary file
            import os
            os.unlink(temp_file_path)

        # Handle stereo audio by converting to mono
        if data.ndim == 2:
            data = np.mean(data, axis=1)

        # Whisper works better with certain sample rates, let's try to resample if needed
        if samplerate != 16000 and samplerate != 44100 and samplerate != 48000:
            # Use torchaudio for resampling if available
            try:
                import torchaudio.transforms as T
                import torch

                # Convert to tensor if it's not already
                if not isinstance(data, torch.Tensor):
                    data_tensor = torch.from_numpy(data).float()

                # Resample to 16000 Hz (Whisper's preferred sample rate)
                resampler = T.Resample(orig_freq=samplerate, new_freq=16000)
                data_tensor = resampler(data_tensor)
                data = data_tensor.numpy()
                samplerate = 16000
                print(f"Resampled audio from {samplerate}Hz to 16000Hz")
            except ImportError:
                print("torchaudio not available for resampling, using original sample rate")

        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        stt_pipeline = get_stt_pipeline()

        # Check if audio is > 30 seconds (30000 mel features for Whisper)
        audio_duration_seconds = len(data) / samplerate
        use_timestamps = audio_duration_seconds > 30

        if use_timestamps:
            print(f"Audio duration: {audio_duration_seconds:.1f}s (> 30s), enabling timestamps for long-form generation")

        result = stt_pipeline(
            {"array": data, "sampling_rate": samplerate},
            return_timestamps=use_timestamps
        )

        # Extract text from result
        transcribed_text = ""
        if isinstance(result, dict):
            transcribed_text = result.get("text", "").strip()
        elif isinstance(result, str):
            transcribed_text = result.strip()
        else:
            # Handle other possible result formats
            transcribed_text = str(result).strip()

        if not transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No speech detected in audio or transcription failed"
            )

        print(f"Transcribed text: {transcribed_text}")
        return JSONResponse(
            content={"text": transcribed_text},
            status_code=status.HTTP_200_OK
        )

    except HTTPException:
        raise
    except Exception as e:
        # Log the error and return a generic error
        print(f"STT processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech-to-text processing failed"
        )


@router.get("/stt/status")
async def get_stt_status():
    """
    Get STT model status
    """
    from app.core.stt_model import (
        get_stt_initialization_state,
        get_stt_initialization_progress,
        get_stt_initialization_error
    )

    state = get_stt_initialization_state()
    progress = get_stt_initialization_progress()
    error = get_stt_initialization_error()

    return {
        "state": state,
        "ready": is_stt_ready(),
        "progress": progress,
        "error": error
    }
