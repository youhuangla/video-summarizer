"""Audio extraction and transcription.

Extracts audio from video using FFmpeg (via pydub),
then transcribes using Whisper HTTP API.
"""

import hashlib
import json
import requests
import subprocess
from pathlib import Path
from typing import Union, List, Dict, Optional

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


WHISPER_API_BASE = "http://127.0.0.1:18181/v1"  # Direct whisper.cpp API
WHISPER_MODEL = "whisper-1"


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
    
    1. Extracts audio from video using FFmpeg (pydub)
    2. Transcribes using Whisper HTTP API
    """
    
    def __init__(
        self,
        api_base: str = WHISPER_API_BASE,
        api_key: str = "",
        cache_dir: str = "./cache",
        model: str = WHISPER_MODEL
    ):
        """Initialize audio extractor.
        
        Args:
            api_base: Base URL for Whisper API
            api_key: API key (optional for local)
            cache_dir: Directory to cache transcription results
            model: Whisper model name
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        
        # Audio extraction cache
        self.audio_cache_dir = Path(cache_dir) / "audio"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_transcript(
        self,
        video_path: Union[str, Path],
        video_id: str = None
    ) -> List[TranscriptSegment]:
        """Extract transcript from video.
        
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
        
        # Check transcript cache
        cache_file = self.cache_dir / f"{video_id}_transcript.json"
        if cache_file.exists():
            print("Loading cached transcript...")
            return self._load_from_cache(cache_file)
        
        # Step 1: Extract audio from video
        print("Extracting audio from video...")
        audio_path = self._extract_audio(video_path, video_id)
        if not audio_path:
            print("Warning: Failed to extract audio")
            return []
        
        # Step 2: Transcribe audio
        print("Transcribing audio...")
        segments = self._call_whisper_api(audio_path)
        
        # Save to cache
        if segments:
            self._save_to_cache(cache_file, segments)
        
        return segments
    
    def _extract_audio(self, video_path: Path, video_id: str) -> Optional[Path]:
        """Extract audio from video file using FFmpeg.
        
        Args:
            video_path: Path to video file
            video_id: Video ID for caching
            
        Returns:
            Path to extracted audio file (WAV format)
        """
        # Check if audio already extracted
        audio_path = self.audio_cache_dir / f"{video_id}.wav"
        if audio_path.exists():
            print(f"Using cached audio: {audio_path}")
            return audio_path
        
        try:
            # Use FFmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz (Whisper optimal)
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                str(audio_path)
            ]
            
            print(f"Running: ffmpeg -i {video_path.name} ...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and audio_path.exists():
                print(f"Audio extracted: {audio_path}")
                return audio_path
            else:
                print(f"FFmpeg error: {result.stderr[:200]}")
                return None
                
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def _call_whisper_api(self, audio_path: Path) -> List[TranscriptSegment]:
        """Call Whisper HTTP API to transcribe audio.
        
        Args:
            audio_path: Path to audio file (WAV)
            
        Returns:
            List of transcript segments
        """
        url = f"{self.api_base}/audio/transcriptions"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {
                "model": self.model,
                "language": "zh",
                "response_format": "verbose_json"
            }
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=600)
                response.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                print(f"Warning: Cannot connect to Whisper API: {e}")
                return []
            except requests.exceptions.Timeout:
                print("Warning: Whisper API request timed out.")
                return []
        
        result = response.json()
        
        # Parse segments
        segments = []
        if "segments" in result and result["segments"]:
            for seg in result["segments"]:
                segments.append(TranscriptSegment(
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", "").strip()
                ))
        elif "text" in result and result["text"]:
            segments.append(TranscriptSegment(
                start=0.0,
                end=0.0,
                text=result["text"].strip()
            ))
        
        return segments
    
    def get_transcript_text(
        self,
        segments: List[TranscriptSegment],
        start_time: float = 0,
        end_time: float = None
    ) -> str:
        """Get transcript text within time range."""
        if end_time is None:
            end_time = float('inf')
        
        texts = []
        for seg in segments:
            if seg.end >= start_time and seg.start <= end_time:
                texts.append(seg.text)
        
        return " ".join(texts)
    
    def _compute_video_id(self, video_path: Path) -> str:
        """Compute unique video ID from file content."""
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
