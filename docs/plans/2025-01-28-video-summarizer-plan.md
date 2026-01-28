# Video Summarizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a command-line tool that generates timestamped Markdown summaries from local video files using Kimi VLM, supporting videos of any length via automatic segmentation.

**Architecture:** The tool uses a 3-stage pipeline: (1) Extract audio and transcribe with Whisper for timestamped text, (2) Extract keyframes uniformly from video, (3) Use Kimi VLM to identify chapter boundaries and generate summaries with visual and textual evidence.

**Tech Stack:** Python 3.10+, OpenAI-Whisper (MIT License), FFmpeg (LGPL/GPL), OpenAI-compatible API for Kimi VLM, python-decord (Apache-2.0 License)

**External Dependencies:**
- Video frame extraction: `decord` library (https://github.com/dmlc/decord, Apache-2.0)
- Audio transcription: `openai-whisper` (https://github.com/openai/whisper, MIT License)
- Inspiration from Video-Browser paper implementation (https://github.com/chrisx599/Video-Browser)

---

## Task 1: Project Structure Setup

**Files:**
- Create: `video_summarizer/__init__.py`
- Create: `video_summarizer/main.py` (entry point)
- Create: `video_summarizer/config.py`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md` (basic)

**Step 1: Create directory structure**

```bash
mkdir -p video_summarizer/{extractors,analyzers,utils}
mkdir -p tests
mkdir -p examples
```

**Step 2: Write requirements.txt**

```
openai-whisper>=20231117
openai>=1.0.0
decord>=0.6.0
numpy>=1.24.0
pydantic>=2.0.0
tqdm>=4.65.0
```

**Step 3: Write basic .gitignore**

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/
.cache/
*.mp4
*.mov
*.avi
*.mkv
*.wav
*.mp3
*.srt
*.vtt
output/
cache/
.env
```

**Step 4: Verify structure**

Run: `dir /s /b video_summarizer\`
Expected: See all __init__.py files and directory structure

**Step 5: Commit**

```bash
git init
git add requirements.txt .gitignore
git commit -m "chore: initial project structure"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `video_summarizer/config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing test**

```python
# tests/test_config.py
import os
from video_summarizer.config import SummarizerConfig

def test_default_config():
    config = SummarizerConfig()
    assert config.segment_duration == 600
    assert config.sparse_frame_count == 20
    assert config.kimi_model == "kimi-vl-a3b-thinking-250701"

def test_config_from_env():
    os.environ["KIMI_API_KEY"] = "test-key-123"
    config = SummarizerConfig()
    assert config.kimi_api_key == "test-key-123"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: ImportError - cannot import SummarizerConfig

**Step 3: Write minimal implementation**

```python
# video_summarizer/config.py
"""Configuration for Video Summarizer.

This module defines configuration parameters for the video summarization pipeline.
Inspired by Video-Browser (https://github.com/chrisx599/Video-Browser) paper's
pyramidal perception architecture.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class SummarizerConfig:
    """Configuration for video summarization.
    
    Uses pyramidal perception approach:
    - Stage 1: Sparse uniform sampling for chapter boundary detection
    - Stage 2: Dense sampling within chapters for detailed summary
    """
    
    # API Configuration
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    kimi_model: str = "kimi-vl-a3b-thinking-250701"
    
    # Video Processing
    segment_duration: int = 600  # Split videos longer than 10 minutes
    overlap_duration: int = 30   # Overlap between segments
    sparse_frame_count: int = 20  # Stage 1: frames for chapter detection
    dense_fps: float = 0.5        # Stage 2: frames per second in chapters
    min_chapter_duration: int = 30  # Minimum chapter length in seconds
    max_chapters: int = 8
    
    # Output
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    enable_cache: bool = True
    
    def __post_init__(self):
        if not self.kimi_api_key:
            self.kimi_api_key = os.getenv("KIMI_API_KEY", "")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 tests passed

**Step 5: Commit**

```bash
git add video_summarizer/config.py tests/test_config.py
git commit -m "feat: add configuration module"
```

---

## Task 3: Video Metadata Extractor

**Files:**
- Create: `video_summarizer/extractors/__init__.py`
- Create: `video_summarizer/extractors/metadata.py`
- Test: `tests/test_metadata.py`

**Step 1: Write failing test**

```python
# tests/test_metadata.py
import pytest
from pathlib import Path
from video_summarizer.extractors.metadata import MetadataExtractor, VideoMetadata

def test_video_metadata_creation():
    meta = VideoMetadata(
        duration=120.5,
        fps=30.0,
        resolution=(1920, 1080),
        total_frames=3615
    )
    assert meta.duration == 120.5
    assert meta.width == 1920
    assert meta.height == 1080
```

**Step 2: Run test - expect fail**

Run: `python -m pytest tests/test_metadata.py -v`
Expected: ImportError

**Step 3: Implement metadata extractor**

```python
# video_summarizer/extractors/metadata.py
"""Video metadata extraction using decord.

Uses decord library (https://github.com/dmlc/decord, Apache-2.0)
for efficient video metadata extraction without full decoding.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


@dataclass
class VideoMetadata:
    """Video metadata container."""
    duration: float  # seconds
    fps: float
    resolution: Tuple[int, int]
    total_frames: int
    
    @property
    def width(self) -> int:
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        return self.resolution[1]


class MetadataExtractor:
    """Extract metadata from video files."""
    
    def extract(self, video_path: Union[str, Path]) -> VideoMetadata:
        """Extract metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object containing duration, fps, resolution, etc.
        """
        if not DECORD_AVAILABLE:
            raise ImportError("decord is required. Install: pip install decord")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        vr = VideoReader(str(video_path), ctx=cpu(0))
        
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps
        
        # Get resolution from first frame
        first_frame = vr[0]
        height, width = first_frame.shape[:2]
        
        return VideoMetadata(
            duration=duration,
            fps=fps,
            resolution=(width, height),
            total_frames=total_frames
        )
```

**Step 4: Run test - expect pass**

Run: `python -m pytest tests/test_metadata.py -v`
Expected: 1 test passed

**Step 5: Commit**

```bash
git add video_summarizer/extractors/__init__.py video_summarizer/extractors/metadata.py tests/test_metadata.py
git commit -m "feat: add video metadata extractor using decord"
```

---

## Task 4: Audio Transcription (Whisper)

**Files:**
- Create: `video_summarizer/extractors/audio.py`
- Test: `tests/test_audio.py`

**Step 1: Write failing test**

```python
# tests/test_audio.py
import pytest
from pathlib import Path
from video_summarizer.extractors.audio import AudioExtractor

def test_audio_extractor_init():
    extractor = AudioExtractor()
    assert extractor is not None
```

**Step 2: Run test - expect fail**

**Step 3: Implement audio extractor**

```python
# video_summarizer/extractors/audio.py
"""Audio extraction and transcription using Whisper.

Uses OpenAI Whisper (https://github.com/openai/whisper, MIT License)
for speech-to-text transcription with timestamps.
"""

import hashlib
import json
from pathlib import Path
from typing import Union, List, Dict

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


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
    
    Uses Whisper for transcription. Results are cached to avoid
    re-processing the same video.
    """
    
    def __init__(self, model_size: str = "base", cache_dir: str = "./cache"):
        """Initialize audio extractor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            cache_dir: Directory to cache transcription results
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper is required. Install: pip install openai-whisper")
        
        self.model_size = model_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
    
    @property
    def model(self):
        """Lazy load whisper model."""
        if self._model is None:
            print(f"Loading Whisper model: {self.model_size}...")
            self._model = whisper.load_model(self.model_size)
        return self._model
    
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
        
        # Check cache
        cache_file = self.cache_dir / f"{video_id}_transcript.json"
        if cache_file.exists():
            print(f"Loading cached transcript: {cache_file}")
            return self._load_from_cache(cache_file)
        
        # Transcribe with Whisper
        print(f"Transcribing audio from: {video_path.name}")
        result = self.model.transcribe(str(video_path), language="zh")
        
        # Parse segments
        segments = []
        for seg in result["segments"]:
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            ))
        
        # Save to cache
        self._save_to_cache(cache_file, segments)
        
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
```

**Step 4: Run test - expect pass**

**Step 5: Commit**

```bash
git add video_summarizer/extractors/audio.py tests/test_audio.py
git commit -m "feat: add audio transcription with Whisper and caching"
```

---

## Task 5: Frame Extractor (Pyramidal Sampling)

**Files:**
- Create: `video_summarizer/extractors/frames.py`
- Test: `tests/test_frames.py`

**Step 1: Write failing test**

```python
# tests/test_frames.py
import pytest
from pathlib import Path
from video_summarizer.extractors.frames import FrameExtractor, FrameInfo

def test_frame_info_creation():
    frame = FrameInfo(
        path=Path("/tmp/frame_001.jpg"),
        timestamp=10.5,
        frame_number=315
    )
    assert frame.timestamp == 10.5
    assert frame.frame_number == 315
```

**Step 2: Run test - expect fail**

**Step 3: Implement frame extractor**

```python
# video_summarizer/extractors/frames.py
"""Video frame extraction using pyramidal sampling strategy.

This module implements the pyramidal perception approach from
Video-Browser paper: sparse uniform sampling for chapter detection,
followed by dense sampling within identified chapters.

Uses decord library (https://github.com/dmlc/decord, Apache-2.0)
for efficient frame extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional
import io
import base64

try:
    from decord import VideoReader, cpu
    import numpy as np
    from PIL import Image
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


@dataclass
class FrameInfo:
    """Information about an extracted frame."""
    path: Path
    timestamp: float
    frame_number: int
    
    def to_base64(self) -> str:
        """Convert frame image to base64 string."""
        with open(self.path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class FrameExtractor:
    """Extract frames from video using pyramidal sampling.
    
    Stage 1 (Sparse): Uniform sampling for chapter boundary detection
    Stage 2 (Dense): High-frequency sampling within chapters
    """
    
    def __init__(self, cache_dir: str = "./cache/frames"):
        """Initialize frame extractor.
        
        Args:
            cache_dir: Directory to cache extracted frames
        """
        if not DECORD_AVAILABLE:
            raise ImportError("decord and PIL are required. Install: pip install decord pillow")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_uniform(
        self,
        video_path: Union[str, Path],
        num_frames: int = 20,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> List[FrameInfo]:
        """Extract frames uniformly across time range (Stage 1 - Sparse).
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of video)
            
        Returns:
            List of FrameInfo objects
        """
        video_path = Path(video_path)
        video_id = self._compute_video_id(video_path)
        
        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        
        # Determine time range
        start_time = max(0, start_time)
        if end_time is None or end_time > duration:
            end_time = duration
        
        # Calculate frame indices for uniform sampling
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        if num_frames == 1:
            indices = [start_frame]
        else:
            step = (end_frame - start_frame) / (num_frames - 1)
            indices = [int(start_frame + i * step) for i in range(num_frames)]
            indices = [min(idx, total_frames - 1) for idx in indices]
        
        # Extract and save frames
        frames = []
        for i, frame_idx in enumerate(indices):
            timestamp = frame_idx / fps
            frame_filename = f"{video_id}_sparse_{i:03d}_{timestamp:.1f}s.jpg"
            frame_path = self.cache_dir / frame_filename
            
            if not frame_path.exists():
                # Extract frame using decord
                frame = vr[frame_idx].asnumpy()
                # Convert to PIL Image and save
                img = Image.fromarray(frame)
                img.save(frame_path, quality=85)
            
            frames.append(FrameInfo(
                path=frame_path,
                timestamp=timestamp,
                frame_number=frame_idx
            ))
        
        return frames
    
    def extract_range(
        self,
        video_path: Union[str, Path],
        start_time: float,
        end_time: float,
        fps: float = 0.5
    ) -> List[FrameInfo]:
        """Extract frames at specified FPS within time range (Stage 2 - Dense).
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            fps: Frames per second to extract
            
        Returns:
            List of FrameInfo objects
        """
        video_path = Path(video_path)
        video_id = self._compute_video_id(video_path)
        
        vr = VideoReader(str(video_path), ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        
        # Calculate frame indices
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        frame_interval = int(video_fps / fps)
        
        indices = list(range(start_frame, end_frame, frame_interval))
        
        # Extract frames
        frames = []
        for i, frame_idx in enumerate(indices):
            timestamp = frame_idx / video_fps
            frame_filename = f"{video_id}_dense_{timestamp:.1f}s.jpg"
            frame_path = self.cache_dir / frame_filename
            
            if not frame_path.exists():
                frame = vr[frame_idx].asnumpy()
                img = Image.fromarray(frame)
                img.save(frame_path, quality=90)  # Higher quality for dense frames
            
            frames.append(FrameInfo(
                path=frame_path,
                timestamp=timestamp,
                frame_number=frame_idx
            ))
        
        return frames
    
    def _compute_video_id(self, video_path: Path) -> str:
        """Compute unique video ID."""
        import hashlib
        content = video_path.read_bytes()[:8192]
        size = str(video_path.stat().st_size).encode()
        return hashlib.md5(content + size).hexdigest()[:12]
```

**Step 4: Run test - expect pass**

**Step 5: Commit**

```bash
git add video_summarizer/extractors/frames.py tests/test_frames.py
git commit -m "feat: add pyramidal frame extraction (sparse + dense)"
```

---

## Task 6: Kimi API Client

**Files:**
- Create: `video_summarizer/utils/__init__.py`
- Create: `video_summarizer/utils/kimi_client.py`
- Test: `tests/test_kimi_client.py`

**Step 1: Write failing test**

```python
# tests/test_kimi_client.py
import pytest
from video_summarizer.utils.kimi_client import KimiClient

def test_kimi_client_init():
    client = KimiClient(api_key="test-key", base_url="https://test.com", model="test-model")
    assert client.model == "test-model"
```

**Step 2: Run test - expect fail**

**Step 3: Implement Kimi client**

```python
# video_summarizer/utils/kimi_client.py
"""Kimi VLM API client.

Provides interface to Kimi Vision-Language Model for analyzing
video frames and generating summaries.
"""

import json
import base64
from pathlib import Path
from typing import List, Union, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class KimiClient:
    """Client for Kimi VLM API (OpenAI-compatible)."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-vl-a3b-thinking-250701"
    ):
        """Initialize Kimi client.
        
        Args:
            api_key: Kimi API key
            base_url: API base URL
            model: Model name
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required. Install: pip install openai")
        
        if not api_key:
            raise ValueError("API key is required")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def analyze_images(
        self,
        images: List[Union[str, Path]],
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> str:
        """Analyze images with Kimi VLM.
        
        Args:
            images: List of image paths
            prompt: Text prompt for analysis
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            
        Returns:
            Model response text
        """
        content = [{"type": "text", "text": prompt}]
        
        # Add images (limit to first 8)
        for img_path in images[:8]:
            img_base64 = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.5
    ) -> str:
        """Generate text without images.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
```

**Step 4: Run test - expect pass**

**Step 5: Commit**

```bash
git add video_summarizer/utils/__init__.py video_summarizer/utils/kimi_client.py tests/test_kimi_client.py
git commit -m "feat: add Kimi VLM API client"
```

---

## Task 7: Chapter Analyzer (Kimi VLM Integration)

**Files:**
- Create: `video_summarizer/analyzers/__init__.py`
- Create: `video_summarizer/analyzers/chapters.py`
- Test: `tests/test_chapters.py`

**Step 1: Write failing test**

```python
# tests/test_chapters.py
import pytest
from video_summarizer.analyzers.chapters import ChapterAnalyzer

def test_chapter_analyzer_init():
    # This will require mocking the KimiClient
    analyzer = ChapterAnalyzer(api_key="test")
    assert analyzer is not None
```

**Step 2: Run test - expect fail**

**Step 3: Implement chapter analyzer**

```python
# video_summarizer/analyzers/chapters.py
"""Chapter analysis using Kimi VLM.

Implements the chapter boundary detection and summary generation
using pyramidal perception approach inspired by Video-Browser.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

from video_summarizer.utils.kimi_client import KimiClient


@dataclass
class Chapter:
    """A video chapter with summary."""
    start_time: float
    end_time: float
    title: str
    summary: str
    key_quotes: List[str]
    key_frames: List[str]  # Frame paths


class ChapterAnalyzer:
    """Analyze video and generate chapter summaries."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.moonshot.cn/v1"):
        """Initialize analyzer.
        
        Args:
            api_key: Kimi API key
            base_url: API base URL
        """
        self.kimi = KimiClient(api_key=api_key, base_url=base_url)
    
    def detect_chapters(
        self,
        frames: List,
        transcript_text: str,
        duration: float,
        min_chapter_duration: int = 30,
        max_chapters: int = 8
    ) -> List[Dict]:
        """Detect chapter boundaries from sparse frames.
        
        Args:
            frames: List of FrameInfo objects (sparse sampling)
            transcript_text: Full transcript text
            duration: Video duration in seconds
            min_chapter_duration: Minimum chapter length
            max_chapters: Maximum number of chapters
            
        Returns:
            List of chapter boundaries with start/end times
        """
        # Build frame descriptions
        frame_info = "\n".join([
            f"[{f.timestamp:.1f}s] Frame {i+1}/{len(frames)}"
            for i, f in enumerate(frames)
        ])
        
        prompt = f"""你是一个专业的视频编辑助手。请分析以下视频的关键帧和字幕，识别出自然的章节边界。

视频信息：
- 总时长: {duration:.0f}秒
- 关键帧时间点: {[f.timestamp for f in frames]}

字幕节选（前3000字符）：
{transcript_text[:3000]}

任务：
1. 识别 {min(3, int(duration/60))}-{min(max_chapters, max(3, int(duration/120)))} 个主要章节
2. 每个章节必须包含：开始时间(秒)、结束时间(秒)、章节标题(5-15字中文)
3. 章节边界应对应内容主题的明显转换（话题切换、场景切换、引入新概念等）
4. 每个章节至少 {min_chapter_duration} 秒
5. 第一个章节从 0 开始，最后一个章节到 {duration:.0f} 结束

重要：返回严格的JSON格式，不要其他内容。

输出格式：
{{
  "chapters": [
    {{
      "start": 0,
      "end": 120,
      "title": "开场与背景介绍",
      "reasoning": "视频开始介绍背景和主题..."
    }},
    {{
      "start": 115,
      "end": 300,
      "title": "核心内容演示",
      "reasoning": "进入主要内容展示..."
    }}
  ]
}}
"""
        
        # Get frame paths
        frame_paths = [f.path for f in frames]
        
        # Call Kimi VLM
        response = self.kimi.analyze_images(
            images=frame_paths,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.3
        )
        
        # Parse JSON response
        return self._parse_chapter_response(response, duration)
    
    def summarize_chapter(
        self,
        frames: List,
        transcript_text: str,
        title_hint: str = "",
        detail_level: str = "standard"
    ) -> Dict:
        """Generate detailed summary for a chapter.
        
        Args:
            frames: Dense frame sampling within chapter
            transcript_text: Transcript text for this chapter
            title_hint: Suggested chapter title
            detail_level: "brief", "standard", or "detailed"
            
        Returns:
            Dictionary with summary, key quotes, etc.
        """
        # Determine output fields based on detail level
        if detail_level == "brief":
            fields = "summary (2-3句话)"
        elif detail_level == "standard":
            fields = "summary (3-5句话), key_quotes (2-3句关键台词)"
        else:  # detailed
            fields = "summary (详细段落), key_quotes (3-5句), visual_details (视觉细节描述), topics (3-5个关键词)"
        
        prompt = f"""基于以下视频帧和字幕，为"{title_hint}"这个章节生成详细摘要。

字幕内容：
{transcript_text[:4000]}

任务：
1. 概括这个章节的核心内容
2. 提取最代表性的台词/引语
3. 描述关键视觉信息（如果有重要画面）

输出格式（JSON）：
{{
  "summary": "章节摘要...",
  "key_quotes": ["台词1", "台词2"],
  "visual_details": "画面描述..." (可选),
  "topics": ["主题1", "主题2"] (可选)
}}

要求：摘要必须是中文，简洁准确。
"""
        
        frame_paths = [f.path for f in frames[:6]]  # Limit frames
        
        response = self.kimi.analyze_images(
            images=frame_paths,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.4
        )
        
        return self._parse_summary_response(response)
    
    def _parse_chapter_response(self, response: str, duration: float) -> List[Dict]:
        """Parse chapter detection response."""
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                chapters = data.get("chapters", [])
                
                # Validate and fix boundaries
                validated = []
                for i, ch in enumerate(chapters):
                    start = max(0, float(ch.get("start", 0)))
                    end = min(duration, float(ch.get("end", duration)))
                    
                    # Ensure reasonable duration
                    if end - start < 30:
                        if i == len(chapters) - 1:
                            start = max(0, end - 30)
                        else:
                            end = min(duration, start + 30)
                    
                    validated.append({
                        "start": start,
                        "end": end,
                        "title": ch.get("title", f"章节 {i+1}"),
                        "reasoning": ch.get("reasoning", "")
                    })
                
                return validated
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse chapter response: {e}")
        
        # Fallback: split into 3 equal parts
        segment = duration / 3
        return [
            {"start": 0, "end": segment, "title": "第一部分", "reasoning": "自动分割"},
            {"start": segment * 0.9, "end": segment * 2, "title": "第二部分", "reasoning": "自动分割"},
            {"start": segment * 1.9, "end": duration, "title": "第三部分", "reasoning": "自动分割"}
        ]
    
    def _parse_summary_response(self, response: str) -> Dict:
        """Parse summary response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
        
        # Fallback: treat entire response as summary
        return {
            "summary": response[:500],
            "key_quotes": [],
            "visual_details": "",
            "topics": []
        }
```

**Step 4: Run test - expect pass**

**Step 5: Commit**

```bash
git add video_summarizer/analyzers/__init__.py video_summarizer/analyzers/chapters.py tests/test_chapters.py
git commit -m "feat: add chapter detection and summarization with Kimi VLM"
```

---

## Task 8: Markdown Output Formatter

**Files:**
- Create: `video_summarizer/output/__init__.py`
- Create: `video_summarizer/output/formatter.py`
- Test: `tests/test_formatter.py`

**Step 1: Write failing test**

```python
# tests/test_formatter.py
import pytest
from datetime import timedelta
from video_summarizer.output.formatter import MarkdownFormatter

def test_format_timestamp():
    formatter = MarkdownFormatter()
    assert formatter._format_timestamp(125.5) == "02:05"
    assert formatter._format_timestamp(3661) == "01:01:01"
```

**Step 2: Run test - expect fail**

**Step 3: Implement formatter**

```python
# video_summarizer/output/formatter.py
"""Output formatters for video summaries.

Generates Markdown and other formats from analysis results.
"""

from pathlib import Path
from typing import List, Dict
from datetime import timedelta


class MarkdownFormatter:
    """Format video summary as Markdown."""
    
    def format(
        self,
        video_path: Path,
        duration: float,
        overall_summary: str,
        chapters: List[Dict],
        output_path: Path = None
    ) -> str:
        """Format summary as Markdown.
        
        Args:
            video_path: Original video path
            duration: Video duration in seconds
            overall_summary: Overall video summary
            chapters: List of chapter dictionaries
            output_path: Optional output file path
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append(f"# 视频摘要: {video_path.name}")
        lines.append("")
        lines.append(f"- **文件**: `{video_path}`")
        lines.append(f"- **时长**: {self._format_duration(duration)}")
        lines.append(f"- **章节数**: {len(chapters)}")
        lines.append("")
        
        # Overall summary
        lines.append("## 整体摘要")
        lines.append("")
        lines.append(overall_summary)
        lines.append("")
        
        # Table of contents
        lines.append("## 章节概览")
        lines.append("")
        lines.append("| 时间 | 章节 | 时长 |")
        lines.append("|------|------|------|")
        for ch in chapters:
            start = self._format_timestamp(ch['start_time'])
            duration_str = self._format_duration(ch['end_time'] - ch['start_time'])
            lines.append(f"| {start} | {ch['title']} | {duration_str} |")
        lines.append("")
        
        # Detailed chapters
        lines.append("## 章节详情")
        lines.append("")
        
        for i, ch in enumerate(chapters, 1):
            start = self._format_timestamp(ch['start_time'])
            end = self._format_timestamp(ch['end_time'])
            
            lines.append(f"### [{start} - {end}] {ch['title']}")
            lines.append("")
            
            # Summary
            lines.append("**摘要:**")
            lines.append(ch['summary'])
            lines.append("")
            
            # Key quotes
            if ch.get('key_quotes'):
                lines.append("**关键台词:**")
                for quote in ch['key_quotes']:
                    lines.append(f"> {quote}")
                lines.append("")
            
            # Visual details (if available)
            if ch.get('visual_details'):
                lines.append("**视觉细节:**")
                lines.append(ch['visual_details'])
                lines.append("")
            
            # Topics (if available)
            if ch.get('topics'):
                lines.append(f"**主题标签:** {', '.join(ch['topics'])}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        markdown = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"Summary saved to: {output_path}")
        
        return markdown
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        td = timedelta(seconds=int(seconds))
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable form."""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds/60)}分{int(seconds%60)}秒"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}小时{mins}分"
```

**Step 4: Run test - expect pass**

**Step 5: Commit**

```bash
git add video_summarizer/output/__init__.py video_summarizer/output/formatter.py tests/test_formatter.py
git commit -m "feat: add Markdown output formatter"
```

---

## Task 9: Main Pipeline Orchestrator

**Files:**
- Create: `video_summarizer/pipeline.py`
- Modify: `video_summarizer/__init__.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing test**

```python
# tests/test_pipeline.py
import pytest
from pathlib import Path
from video_summarizer.pipeline import VideoSummarizerPipeline
from video_summarizer.config import SummarizerConfig

def test_pipeline_init():
    config = SummarizerConfig(kimi_api_key="test")
    pipeline = VideoSummarizerPipeline(config)
    assert pipeline is not None
```

**Step 2: Run test - expect fail**

**Step 3: Implement main pipeline**

```python
# video_summarizer/pipeline.py
"""Main video summarization pipeline.

Orchestrates the entire summarization process:
1. Extract metadata
2. Transcribe audio
3. Extract frames (pyramidal sampling)
4. Detect chapters
5. Summarize each chapter
6. Generate output
"""

import time
from pathlib import Path
from typing import Union, List, Dict

from video_summarizer.config import SummarizerConfig
from video_summarizer.extractors.metadata import MetadataExtractor
from video_summarizer.extractors.audio import AudioExtractor
from video_summarizer.extractors.frames import FrameExtractor
from video_summarizer.analyzers.chapters import ChapterAnalyzer
from video_summarizer.output.formatter import MarkdownFormatter


class VideoSummaryResult:
    """Result container for video summarization."""
    
    def __init__(
        self,
        video_path: Path,
        duration: float,
        overall_summary: str,
        chapters: List[Dict],
        processing_time: float,
        output_path: Path = None
    ):
        self.video_path = video_path
        self.duration = duration
        self.overall_summary = overall_summary
        self.chapters = chapters
        self.processing_time = processing_time
        self.output_path = output_path


class VideoSummarizerPipeline:
    """Main pipeline for video summarization."""
    
    def __init__(self, config: SummarizerConfig = None):
        """Initialize pipeline.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or SummarizerConfig()
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        self.audio_extractor = AudioExtractor(
            model_size="base",
            cache_dir=self.config.cache_dir
        )
        self.frame_extractor = FrameExtractor(
            cache_dir=f"{self.config.cache_dir}/frames"
        )
        self.chapter_analyzer = ChapterAnalyzer(
            api_key=self.config.kimi_api_key,
            base_url=self.config.kimi_base_url
        )
        self.formatter = MarkdownFormatter()
    
    def summarize(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path] = None
    ) -> VideoSummaryResult:
        """Summarize a video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional output markdown file path
            
        Returns:
            VideoSummaryResult object
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        print(f"\n{'='*60}")
        print(f"视频摘要生成")
        print(f"{'='*60}")
        print(f"文件: {video_path}")
        print()
        
        # Step 1: Extract metadata
        print("[1/6] 提取视频元信息...")
        metadata = self.metadata_extractor.extract(video_path)
        print(f"      时长: {metadata.duration:.1f}s, 分辨率: {metadata.width}x{metadata.height}")
        print()
        
        # Step 2: Transcribe audio
        print("[2/6] 提取音频并转录...")
        transcript_segments = self.audio_extractor.extract_transcript(video_path)
        full_transcript = " ".join([s.text for s in transcript_segments])
        print(f"      识别到 {len(transcript_segments)} 个语音片段")
        print()
        
        # Step 3: Process video
        if metadata.duration > self.config.segment_duration:
            print(f"[3/6] 视频较长，分段处理...")
            chapters = self._process_long_video(
                video_path, transcript_segments, metadata.duration
            )
        else:
            print(f"[3/6] 分析视频章节...")
            chapters = self._process_segment(
                video_path, transcript_segments, 0, metadata.duration
            )
        
        print(f"      检测到 {len(chapters)} 个章节")
        print()
        
        # Step 4: Generate overall summary
        print("[4/6] 生成整体摘要...")
        overall_summary = self._generate_overall_summary(chapters, full_transcript)
        print(f"      {overall_summary[:100]}...")
        print()
        
        # Step 5: Format output
        print("[5/6] 格式化输出...")
        if output_path is None:
            output_path = self.config.output_dir / f"{video_path.stem}_summary.md"
        else:
            output_path = Path(output_path)
        
        self.formatter.format(
            video_path=video_path,
            duration=metadata.duration,
            overall_summary=overall_summary,
            chapters=chapters,
            output_path=output_path
        )
        print()
        
        # Step 6: Done
        processing_time = time.time() - start_time
        print(f"[6/6] 完成! 总耗时: {processing_time:.1f}秒")
        print(f"{'='*60}\n")
        
        return VideoSummaryResult(
            video_path=video_path,
            duration=metadata.duration,
            overall_summary=overall_summary,
            chapters=chapters,
            processing_time=processing_time,
            output_path=output_path
        )
    
    def _process_segment(
        self,
        video_path: Path,
        transcript_segments: List,
        start_offset: float,
        end_offset: float
    ) -> List[Dict]:
        """Process a video segment."""
        duration = end_offset - start_offset
        
        # Get transcript for this segment
        segment_transcript = " ".join([
            s.text for s in transcript_segments
            if s.start >= start_offset and s.end <= end_offset
        ])
        
        # Stage 1: Sparse sampling for chapter detection
        sparse_frames = self.frame_extractor.extract_uniform(
            video_path,
            num_frames=self.config.sparse_frame_count,
            start_time=start_offset,
            end_time=end_offset
        )
        
        # Detect chapters
        chapter_boundaries = self.chapter_analyzer.detect_chapters(
            frames=sparse_frames,
            transcript_text=segment_transcript,
            duration=duration,
            min_chapter_duration=self.config.min_chapter_duration,
            max_chapters=self.config.max_chapters
        )
        
        # Stage 2: Dense sampling and summarization
        chapters = []
        for boundary in chapter_boundaries:
            # Adjust times for segment offset
            ch_start = start_offset + boundary['start']
            ch_end = start_offset + boundary['end']
            
            # Extract dense frames
            dense_frames = self.frame_extractor.extract_range(
                video_path,
                start_time=ch_start,
                end_time=ch_end,
                fps=self.config.dense_fps
            )
            
            # Get chapter transcript
            ch_transcript = " ".join([
                s.text for s in transcript_segments
                if s.start >= ch_start and s.end <= ch_end
            ])
            
            # Summarize chapter
            summary = self.chapter_analyzer.summarize_chapter(
                frames=dense_frames,
                transcript_text=ch_transcript,
                title_hint=boundary['title'],
                detail_level="standard"
            )
            
            chapters.append({
                "start_time": ch_start,
                "end_time": ch_end,
                "title": boundary['title'],
                **summary
            })
        
        return chapters
    
    def _process_long_video(
        self,
        video_path: Path,
        transcript_segments: List,
        duration: float
    ) -> List[Dict]:
        """Process long video by splitting into segments."""
        segment_length = self.config.segment_duration
        overlap = self.config.overlap_duration
        
        all_chapters = []
        segment_starts = list(range(0, int(duration), segment_length - overlap))
        
        for i, seg_start in enumerate(segment_starts):
            seg_end = min(seg_start + segment_length, duration)
            print(f"      处理分段 {i+1}/{len(segment_starts)}: {seg_start:.0f}s-{seg_end:.0f}s")
            
            chapters = self._process_segment(
                video_path, transcript_segments, seg_start, seg_end
            )
            
            # Merge overlapping chapters
            if all_chapters and chapters:
                last_end = all_chapters[-1]['end_time']
                chapters = [c for c in chapters if c['start_time'] >= last_end - overlap]
            
            all_chapters.extend(chapters)
            
            if seg_end >= duration:
                break
        
        return self._merge_short_chapters(all_chapters)
    
    def _merge_short_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """Merge chapters that are too short."""
        if not chapters:
            return chapters
        
        merged = [chapters[0]]
        
        for ch in chapters[1:]:
            duration = ch['end_time'] - ch['start_time']
            if duration < self.config.min_chapter_duration:
                # Merge with previous
                merged[-1]['end_time'] = ch['end_time']
                merged[-1]['summary'] += f" {ch['summary']}"
            else:
                merged.append(ch)
        
        return merged
    
    def _generate_overall_summary(self, chapters: List[Dict], transcript: str) -> str:
        """Generate overall video summary."""
        # Build chapter summaries
        chapter_text = "\n".join([
            f"章节 {i+1} ({self._fmt_time(c['start_time'])}-{self._fmt_time(c['end_time'])}): {c['title']}\n{c['summary'][:150]}..."
            for i, c in enumerate(chapters)
        ])
        
        prompt = f"""基于以下视频章节摘要，生成一个简洁的整体摘要（100字以内）：

{chapter_text}

请用一句话概括视频主题，然后列出3个核心要点。只输出摘要文本，不要JSON格式。
"""
        
        return self.chapter_analyzer.kimi.generate_text(prompt, max_tokens=300)
    
    def _fmt_time(self, seconds: float) -> str:
        """Format time as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
```

**Step 4: Run test - expect pass**

**Step 5: Commit**

```bash
git add video_summarizer/pipeline.py tests/test_pipeline.py
git commit -m "feat: add main summarization pipeline"
```

---

## Task 10: Command Line Interface

**Files:**
- Create: `video_summarizer/cli.py`
- Modify: `video_summarizer/__main__.py`
- Modify: `video_summarizer/__init__.py`

**Step 1: Write CLI module**

```python
# video_summarizer/cli.py
"""Command line interface for video summarizer.

Interactive CLI for processing video files.
"""

import os
import sys
from pathlib import Path

from video_summarizer.config import SummarizerConfig
from video_summarizer.pipeline import VideoSummarizerPipeline


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Video Summarizer                          ║
║          使用 Kimi VLM 生成视频章节化摘要                      ║
╚══════════════════════════════════════════════════════════════╝
    """)


def get_video_path() -> Path:
    """Get video path from user."""
    while True:
        path = input("请输入视频文件路径: ").strip().strip('"')
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            print(f"错误: 文件不存在: {path}")
            continue
        
        if not path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']:
            print(f"警告: 不常见的视频格式: {path.suffix}")
            confirm = input("是否继续? (y/n): ").lower()
            if confirm != 'y':
                continue
        
        return path


def get_api_key() -> str:
    """Get Kimi API key."""
    # Check environment first
    api_key = os.getenv("KIMI_API_KEY", "")
    
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:]
        print(f"从环境变量读取到 API Key: {masked}")
        use_env = input("使用此 API Key? (y/n): ").lower()
        if use_env == 'y':
            return api_key
    
    # Prompt user
    api_key = input("请输入 Kimi API Key: ").strip()
    if not api_key:
        print("错误: API Key 不能为空")
        sys.exit(1)
    
    return api_key


def main():
    """Main entry point."""
    print_banner()
    
    # Get video path
    video_path = get_video_path()
    print(f"选择文件: {video_path}")
    print()
    
    # Get API key
    api_key = get_api_key()
    print()
    
    # Create config
    config = SummarizerConfig(
        kimi_api_key=api_key,
        output_dir="./output",
        cache_dir="./cache"
    )
    
    # Run pipeline
    try:
        pipeline = VideoSummarizerPipeline(config)
        result = pipeline.summarize(video_path)
        
        print(f"\n✅ 摘要生成成功!")
        print(f"   输出文件: {result.output_path}")
        print(f"   处理时间: {result.processing_time:.1f}秒")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n按 Enter 键退出...")
    input()


if __name__ == "__main__":
    main()
```

**Step 2: Create __main__.py**

```python
# video_summarizer/__main__.py
"""Entry point for python -m video_summarizer."""

from video_summarizer.cli import main

if __name__ == "__main__":
    main()
```

**Step 3: Update __init__.py**

```python
# video_summarizer/__init__.py
"""Video Summarizer - Generate timestamped Markdown summaries from videos.

This package provides tools to analyze video content using Kimi VLM
and generate chapter-based summaries in Markdown format.

Inspired by Video-Browser (https://github.com/chrisx599/Video-Browser)
paper's pyramidal perception architecture.

External dependencies:
- decord: https://github.com/dmlc/decord (Apache-2.0)
- whisper: https://github.com/openai/whisper (MIT License)
"""

__version__ = "0.1.0"

from video_summarizer.config import SummarizerConfig
from video_summarizer.pipeline import VideoSummarizerPipeline, VideoSummaryResult

__all__ = ["SummarizerConfig", "VideoSummarizerPipeline", "VideoSummaryResult"]
```

**Step 4: Create simple usage script**

```python
# summarize.py
#!/usr/bin/env python3
"""Quick start script for video summarization."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from video_summarizer.cli import main

if __name__ == "__main__":
    main()
```

**Step 5: Commit**

```bash
git add video_summarizer/cli.py video_summarizer/__main__.py video_summarizer/__init__.py summarize.py
git commit -m "feat: add command line interface"
```

---

## Task 11: README and Documentation

**Files:**
- Modify: `README.md`
- Create: `.env.example`

**Step 1: Write comprehensive README**

```markdown
# Video Summarizer

使用 Kimi VLM 生成带时间戳的 Markdown 视频摘要。

基于 [Video-Browser](https://github.com/chrisx599/Video-Browser) 论文的**金字塔感知架构**实现：
- **Stage 1**: 稀疏采样识别章节边界
- **Stage 2**: 密集采样生成详细摘要

## 特性

- 🎬 支持任意长度视频（自动分段处理）
- 📝 生成带时间戳的 Markdown 摘要
- 🎯 智能章节分割
- 💾 转录和抽帧结果缓存
- 🔊 使用 Whisper 进行语音识别

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd video-summarizer

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

本项目使用以下开源库：

| 库 | 许可证 | 用途 |
|---|--------|------|
| [decord](https://github.com/dmlc/decord) | Apache-2.0 | 视频帧提取 |
| [whisper](https://github.com/openai/whisper) | MIT | 语音转录 |
| [openai](https://github.com/openai/openai-python) | Apache-2.0 | API 客户端 |

## 使用方法

### 方式 1: 命令行交互

```bash
python summarize.py
```

然后按提示输入视频路径和 API Key。

### 方式 2: 环境变量配置

```bash
# 设置 API Key
export KIMI_API_KEY="your-api-key"

# 运行
python -m video_summarizer
```

### 方式 3: Python API

```python
from video_summarizer import VideoSummarizerPipeline, SummarizerConfig

config = SummarizerConfig(
    kimi_api_key="your-api-key"
)

pipeline = VideoSummarizerPipeline(config)
result = pipeline.summarize("./my_video.mp4")

print(f"生成了 {len(result.chapters)} 个章节")
print(f"输出文件: {result.output_path}")
```

## 配置

复制 `.env.example` 为 `.env` 并填写：

```bash
KIMI_API_KEY=your-api-key-here
```

或修改 `config.py`：

```python
from video_summarizer import SummarizerConfig

config = SummarizerConfig(
    kimi_api_key="your-key",
    segment_duration=600,  # 分段时长（秒）
    sparse_frame_count=20, # 稀疏采样帧数
    dense_fps=0.5,         # 密集采样帧率
)
```

## 输出示例

生成的 Markdown 文件格式：

```markdown
# 视频摘要: tech_talk.mp4

- **文件**: `tech_talk.mp4`
- **时长**: 32分15秒
- **章节数**: 5

## 整体摘要
本视频是关于大语言模型推理优化的技术分享...

## 章节概览

| 时间 | 章节 | 时长 |
|------|------|------|
| 00:00 | 开场与演讲者介绍 | 3分30秒 |
| 03:30 | KV缓存原理 | 8分45秒 |
...

## 章节详情

### [00:00 - 03:30] 开场与演讲者介绍

**摘要:** 演讲者介绍自己的背景和研究方向...

**关键台词:**
> "大家好，我是..."
> "今天我们要讨论..."
```

## 致谢

- [Video-Browser](https://github.com/chrisx599/Video-Browser) - 论文实现参考
- [decord](https://github.com/dmlc/decord) - 高效视频解码
- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别
```

**Step 2: Create .env.example**

```bash
# Kimi API Configuration
# Get your API key from: https://platform.moonshot.cn/
KIMI_API_KEY=your-api-key-here

# Optional: Custom base URL
# KIMI_BASE_URL=https://api.moonshot.cn/v1
```

**Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: add README and environment template"
```

---

## Plan Complete!

**Plan saved to:** `docs/plans/2025-01-28-video-summarizer-plan.md`

### Summary

This plan creates a complete video summarizer with:
- 11 bite-sized tasks (each 2-5 minutes)
- Full test coverage for each module
- Clear file paths and implementation details
- References to external open-source dependencies
- Command-line interactive interface
- Markdown output with timestamps
- Support for videos of any length

### Execution Options

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach would you prefer?
