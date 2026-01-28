"""Video frame extraction using pyramidal sampling strategy.

This module implements the pyramidal perception approach from
Video-Browser paper: sparse uniform sampling for chapter detection,
followed by dense sampling within identified chapters.

Uses decord library (https://github.com/dmlc/decord, Apache-2.0)
for efficient frame extraction.
"""

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional

try:
    from decord import VideoReader, cpu
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
    
    def __init__(self, cache_dir: str = "./cache/frames", resize_to: tuple = None):
        """Initialize frame extractor.
        
        Args:
            cache_dir: Directory to cache extracted frames
            resize_to: Optional (width, height) to resize frames for VRAM saving
        """
        if not DECORD_AVAILABLE:
            raise ImportError("decord and PIL are required. Install: uv pip install decord pillow")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resize_to = resize_to  # (width, height) or None for original size
    
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
        duration = total_frames / fps if fps > 0 else 0
        
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
            timestamp = frame_idx / fps if fps > 0 else 0
            frame_filename = f"{video_id}_sparse_{i:03d}_{timestamp:.1f}s.jpg"
            frame_path = self.cache_dir / frame_filename
            
            if not frame_path.exists():
                # Extract frame using decord
                frame = vr[frame_idx].asnumpy()
                # Convert to PIL Image and resize if needed
                img = Image.fromarray(frame)
                if self.resize_to:
                    img = img.resize(self.resize_to, Image.Resampling.LANCZOS)
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
        
        if video_fps <= 0:
            return []
        
        # Calculate frame indices
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        frame_interval = max(1, int(video_fps / fps))
        
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
                if self.resize_to:
                    img = img.resize(self.resize_to, Image.Resampling.LANCZOS)
                img.save(frame_path, quality=90)  # Higher quality for dense frames
            
            frames.append(FrameInfo(
                path=frame_path,
                timestamp=timestamp,
                frame_number=frame_idx
            ))
        
        return frames
    
    def _compute_video_id(self, video_path: Path) -> str:
        """Compute unique video ID."""
        content = video_path.read_bytes()[:8192]
        size = str(video_path.stat().st_size).encode()
        return hashlib.md5(content + size).hexdigest()[:12]
