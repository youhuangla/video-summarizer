"""Audio extraction and transcription using Whisper.

Supports:
1. Local Whisper HTTP API (Whisper.cpp Server)
2. Direct whisper library (fallback)
"""

import hashlib
import json
import requests
from pathlib import Path
from typing import Union, List, Dict, Optional

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# Default Whisper API (OpenAI compatible)
WHISPER_API_BASE = "http://127.0.0.1:8281/v1"  # NovaAI Whisper API
WHISPER_MODEL = "whisper-1"
WHISPER_API_KEY = "novaai"


class TranscriptSegment:
    """A single transcript segment with timestamp."""
    
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text
    
    def __repr__(self):
        return f"[{self.start:.1f}s-{self.end:.1f}s] {self.text[:50]}..."


class AudioExtractor:
    """Extract and transcribe audio from video files.
    
    Tries HTTP API first, falls back to local whisper library.
    Results are cached to avoid re-processing the same video.
    """
    
    def __init__(
        self,
        api_base: str = WHISPER_API_BASE,
        api_key: str = WHISPER_API_KEY,
        cache_dir: str = "./cache",
        model: str = WHISPER_MODEL
    ):
        """Initialize audio extractor.
        
        Args:
            api_base: Base URL for Whisper API (OpenAI compatible)
            api_key: API key for Whisper API
            cache_dir: Directory to cache transcription results
            model: Whisper model to use
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._whisper_model = None
    
    @property
    def whisper_model(self):
        """Lazy load whisper model."""
        if self._whisper_model is None and WHISPER_AVAILABLE:
            print(f"Loading Whisper model: {self.model}...")
            self._whisper_model = whisper.load_model(self.model)
        return self._whisper_model
    
    def extract_transcript(
        self,
        video_path: Union[str, Path],
        video_id: str = None
    ) -> List[TranscriptSegment]:
        """Extract transcript from video using local Whisper API.
        
        Args:
            video_path: Path to video file
            video_id: Unique video ID for caching (optional)
            
        Returns:
            List of TranscriptSegment objects with timestamps
        """
        video_path = Path(video_path)
        
        # Generate video ID from file if not provided
        if video_id is None:
            video_id = self._compute_video_id(video_path)
        
        # Check cache
        cache_file = self.cache_dir / f"{video_id}_transcript.json"
        if cache_file.exists():
            print(f"Loading cached transcript...")
            return self._load_from_cache(cache_file)
        
        # Try HTTP API first, then fall back to local whisper
        segments = self._call_whisper_api(video_path)
        
        # If API failed or returned empty, try local whisper
        if not segments and WHISPER_AVAILABLE:
            print("API returned empty result, trying local whisper...")
            segments = self._call_local_whisper(video_path)
        
        # Save to cache
        self._save_to_cache(cache_file, segments)
        
        return segments
    
    def _call_whisper_api(self, video_path: Path) -> List[TranscriptSegment]:
        """Call Whisper API (OpenAI compatible) to transcribe video.
        
        API Endpoint: POST /v1/audio/transcriptions
        Compatible with NovaAI Whisper API and OpenAI API.
        """
        # Try NovaAI proxy first, then fallback to direct whisper.cpp
        if self.api_base == "http://127.0.0.1:8281/v1":
            # NovaAI proxy forwards to http://127.0.0.1:18181
            # Use direct whisper.cpp endpoint
            url = "http://127.0.0.1:18181/v1/audio/transcriptions"
        else:
            url = f"{self.api_base}/audio/transcriptions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        with open(video_path, "rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            data = {
                "model": self.model,
                "language": "zh",
                "response_format": "verbose_json"
            }
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=600)
                response.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                print(f"Warning: Cannot connect to Whisper API at {self.api_base}: {e}")
                return []
            except requests.exceptions.Timeout:
                print("Warning: Whisper API request timed out.")
                return []
        
        result = response.json()
        
        # Parse result (OpenAI compatible format)
        segments = []
        
        # Check if result has segments (verbose format)
        if "segments" in result and result["segments"]:
            for seg in result["segments"]:
                segments.append(TranscriptSegment(
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", "").strip()
                ))
        # Check if result has text field (simple format)
        elif "text" in result and result["text"]:
            segments.append(TranscriptSegment(
                start=0.0,
                end=0.0,
                text=result["text"].strip()
            ))
        
        return segments
    
    def _call_local_whisper(self, video_path: Path) -> List[TranscriptSegment]:
        """Call local whisper library to transcribe video.
        
        Fallback when HTTP API fails.
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper library not available. Install: uv pip install openai-whisper")
        
        print(f"Transcribing with local whisper model: {self.model}")
        result = self.whisper_model.transcribe(str(video_path), language="zh")
        
        # Parse segments
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip()
            ))
        
        return segments
    
    def get_transcript_text(
        self,
        segments: List[TranscriptSegment],
        start_time: float = 0,
        end_time: float = None
    ) -> str:
        """Get transcript text within time range.
        
        Args:
            segments: List of transcript segments
            start_time: Start time in seconds
            end_time: End time in seconds (None for all)
            
        Returns:
            Concatenated transcript text
        """
        if end_time is None:
            end_time = float('inf')
        
        texts = []
        for seg in segments:
            if seg.end >= start_time and seg.start <= end_time:
                texts.append(seg.text)
        
        return " ".join(texts)
    
    def _compute_video_id(self, video_path: Path) -> str:
        """Compute unique video ID from file content."""
        # Use first 8KB + file size as hash input
        content = video_path.read_bytes()[:8192]
        size = str(video_path.stat().st_size).encode()
        return hashlib.md5(content + size).hexdigest()[:12]
    
    def _save_to_cache(self, cache_file: Path, segments: List[TranscriptSegment]):
        """Save transcript to cache."""
        data = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_from_cache(self, cache_file: Path) -> List[TranscriptSegment]:
        """Load transcript from cache."""
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TranscriptSegment(d["start"], d["end"], d["text"]) for d in data]
