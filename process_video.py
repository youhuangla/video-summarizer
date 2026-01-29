#!/usr/bin/env python3
"""处理指定视频并生成摘要。"""
import sys
import os

# 强制 UTF-8
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from video_summarizer import VideoSummarizerPipeline, SummarizerConfig

# 配置
config = SummarizerConfig()
pipeline = VideoSummarizerPipeline(config)

# 视频路径
video_path = r"C:\diy_tools\github\yutto-skill\downloads\微信聊天记录工具调研分享.mp4"

print(f"[启动] 处理视频: {os.path.basename(video_path)}")
print(f"[配置] VLM: {config.model}")
print(f"[配置] API: {config.base_url}")

try:
    result = pipeline.summarize(video_path)
    print(f"\n[完成] 输出: {result.output_path}")
    print(f"[完成] 章节数: {len(result.chapters) if result.chapters else 0}")
    if result.chapters:
        print("\n章节概览:")
        for ch in result.chapters:
            start_mins = int(ch['start_time'] // 60)
            start_secs = int(ch['start_time'] % 60)
            end_mins = int(ch['end_time'] // 60)
            end_secs = int(ch['end_time'] % 60)
            print(f"  [{start_mins:02d}:{start_secs:02d}-{end_mins:02d}:{end_secs:02d}] {ch['title']}")
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
