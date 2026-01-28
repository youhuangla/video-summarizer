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
            
        Raises:
            ImportError: If decord is not installed
            FileNotFoundError: If video file does not exist
        """
        if not DECORD_AVAILABLE:
            raise ImportError("decord is required. Install: uv pip install decord")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        vr = VideoReader(str(video_path), ctx=cpu(0))
        
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps if fps > 0 else 0
        
        # Get resolution from first frame
        first_frame = vr[0]
        height, width = first_frame.shape[:2]
        
        return VideoMetadata(
            duration=duration,
            fps=fps,
            resolution=(width, height),
            total_frames=total_frames
        )
