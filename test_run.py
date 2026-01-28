#!/usr/bin/env python3
"""Test run with second video."""

import glob
from pathlib import Path
from video_summarizer import VideoSummarizerPipeline, SummarizerConfig

# 找到视频文件
files = glob.glob('C:/diy_tools/github/yutto-skill/downloads/*.mp4')
if not files:
    print('No video files found')
    exit(1)

# 用第二个视频（如果存在），否则用第一个
video_path = files[0] if len(files) == 1 else files[1]
print('Processing video...')
print('='*60)

config = SummarizerConfig(
    api_key='EMPTY',
    base_url='http://127.0.0.1:1234/v1',
    model='GLM46V-Flash',
    output_dir='./output',
    cache_dir='./cache'
)

pipeline = VideoSummarizerPipeline(config)
result = pipeline.summarize(video_path)

print(f'\nCompleted!')
print(f'Chapters: {len(result.chapters)}')
print(f'Time: {result.processing_time:.1f}s')
print(f'Output: {result.output_path}')
