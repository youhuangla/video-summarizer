#!/usr/bin/env python3
"""Test fallback pipeline."""

import glob
from pathlib import Path
from video_summarizer import FallbackPipeline, SummarizerConfig

# Find video
files = glob.glob('C:/diy_tools/github/yutto-skill/downloads/*.mp4')
if not files:
    print('No video found')
    exit(1)

video_path = Path(files[0])
print(f'Testing fallback mode...')

config = SummarizerConfig()
pipeline = FallbackPipeline(config)
result = pipeline.process(video_path)

print(f'\n=== Results ===')
print(f'Transcript segments: {result.transcript_segments}')
print(f'Processing time: {result.processing_time:.1f}s')
print('\nOutput files:')
for key, path in result.output_files.items():
    if key != 'segment_files':
        safe_path = str(path).encode('ascii', 'ignore').decode()
        print(f'  {key}: {safe_path}')
