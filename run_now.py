#!/usr/bin/env python3
import sys
import os
# 强制 UTF-8
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from video_summarizer import VideoSummarizerPipeline, SummarizerConfig

config = SummarizerConfig()
pipeline = VideoSummarizerPipeline(config)

video_path = r"C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4"
print(f"[启动] 处理视频: {os.path.basename(video_path)}")
print(f"[配置] VLM: {config.model}")
print(f"[配置] API: {config.base_url}")

try:
    result = pipeline.summarize(video_path)
    print(f"\n[完成] 输出: {result.output_path}")
    print(f"[完成] 章节数: {len(result.chapters) if result.chapters else 0}")
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
