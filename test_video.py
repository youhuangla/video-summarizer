#!/usr/bin/env python3
"""测试视频摘要 - 带详细日志"""
import logging
import sys
import time

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from video_summarizer import VideoSummarizerPipeline, SummarizerConfig

def main():
    config = SummarizerConfig()
    pipeline = VideoSummarizerPipeline(config)

    print(f"=" * 60)
    print(f"配置信息:")
    print(f"  API Key: {'已设置' if config.api_key else '未设置(使用本地)'}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Model: {config.model}")
    print(f"  Sparse Frames: {config.sparse_frame_count}")
    print(f"  Dense FPS: {config.dense_fps}")
    print(f"=" * 60)

    video_path = r"C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4"
    print(f"\n开始处理: {video_path}")
    print(f"开始时间: {time.strftime('%H:%M:%S')}")
    
    start = time.time()
    try:
        result = pipeline.summarize(video_path)
        elapsed = time.time() - start
        
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
        print(f"章节数: {len(result.chapters) if result.chapters else 0}")
        if result.transcript:
            print(f"转录片段: {len(result.transcript)}")
        if result.output_file:
            print(f"输出文件: {result.output_file}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
