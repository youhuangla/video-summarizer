"""Configuration for Video Summarizer.

This module defines configuration parameters for the video summarization pipeline.
Inspired by Video-Browser (https://github.com/chrisx599/Video-Browser) paper's
pyramidal perception architecture.
"""

import os
from dataclasses import dataclass
from pathlib import Path
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
        
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
