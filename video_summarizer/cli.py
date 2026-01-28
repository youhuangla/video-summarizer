"""Command line interface for video summarizer.

Interactive CLI for processing video files.
"""

import os
import sys
from pathlib import Path

from video_summarizer.config import SummarizerConfig
from video_summarizer.pipeline import VideoSummarizerPipeline


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Video Summarizer                          ║
║          使用 Kimi VLM 生成视频章节化摘要                      ║
╚══════════════════════════════════════════════════════════════╝
    """)


def get_video_path() -> Path:
    """Get video path from user."""
    while True:
        path = input("请输入视频文件路径: ").strip().strip('"')
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            print(f"错误: 文件不存在: {path}")
            continue
        
        if not path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm']:
            print(f"警告: 不常见的视频格式: {path.suffix}")
            confirm = input("是否继续? (y/n): ").lower()
            if confirm != 'y':
                continue
        
        return path


def get_api_key() -> str:
    """Get API key (for local API, can use EMPTY)."""
    # Check environment first
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
        print(f"从环境变量读取到 API Key: {masked}")
        use_env = input("使用此 API Key? (y/n, 或直接回车跳过): ").lower()
        if use_env == 'y' or use_env == '':
            return api_key
    
    # For local API, default to EMPTY
    default_key = "EMPTY"
    api_key = input(f"请输入 API Key (直接回车使用 '{default_key}' 用于本地API): ").strip()
    if not api_key:
        api_key = default_key
    
    return api_key


def main():
    """Main entry point."""
    print_banner()
    
    # Get video path
    video_path = get_video_path()
    print(f"选择文件: {video_path}")
    print()
    
    # Get API key
    api_key = get_api_key()
    print()
    
    # Create config
    config = SummarizerConfig(
        kimi_api_key=api_key,
        output_dir="./output",
        cache_dir="./cache"
    )
    
    # Run pipeline
    try:
        pipeline = VideoSummarizerPipeline(config)
        result = pipeline.summarize(video_path)
        
        print(f"\n✅ 摘要生成成功!")
        print(f"   输出文件: {result.output_path}")
        print(f"   处理时间: {result.processing_time:.1f}秒")
        
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n按 Enter 键退出...")
    input()


if __name__ == "__main__":
    main()
