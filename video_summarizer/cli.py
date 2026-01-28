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
    """Get Kimi API key."""
    # Check environment first
    api_key = os.getenv("KIMI_API_KEY", "")
    
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:]
        print(f"从环境变量读取到 API Key: {masked}")
        use_env = input("使用此 API Key? (y/n): ").lower()
        if use_env == 'y':
            return api_key
    
    # Prompt user
    api_key = input("请输入 Kimi API Key: ").strip()
    if not api_key:
        print("错误: API Key 不能为空")
        sys.exit(1)
    
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
