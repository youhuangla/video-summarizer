#!/usr/bin/env python3
"""Test audio extraction and transcription."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pathlib import Path
from video_summarizer.extractors.audio import AudioExtractor

video_path = Path(r'C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4')

print("=" * 60)
print("Testing Audio Extraction + Transcription")
print("=" * 60)

extractor = AudioExtractor(
    api_base="http://127.0.0.1:18181/v1",
    cache_dir="./cache"
)

segments = extractor.extract_transcript(video_path)

print(f"\nFound {len(segments)} segments")
if segments:
    print("\nFirst 5 segments:")
    for seg in segments[:5]:
        print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
