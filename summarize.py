#!/usr/bin/env python3
"""Quick start script for video summarization."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from video_summarizer.cli import main

if __name__ == "__main__":
    main()
