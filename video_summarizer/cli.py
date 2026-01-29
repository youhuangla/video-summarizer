"""Command line interface for video summarizer.

Interactive CLI for processing video files.
Supports two modes:
- Full mode: Extract audio + VLM analysis + Generate summary
- Fallback mode: Extract audio only + Export transcripts
"""

import os
import sys
from pathlib import Path

from video_summarizer.config import SummarizerConfig
from video_summarizer.pipeline import VideoSummarizerPipeline
from video_summarizer.fallback_pipeline import FallbackPipeline, check_vlm_available


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Video Summarizer                          ║
║          视频转录与章节化摘要生成工具                          ║
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


def select_mode(config: SummarizerConfig) -> str:
    """Select processing mode.
    
    Returns:
        'full' for VLM analysis mode, 'fallback' for transcript only mode
    """
    print("\n选择处理模式:")
    print("  1. 完整模式 - 语音转录 + AI分析生成章节摘要 (需要大模型服务)")
    print("  2. 转录模式 - 仅语音转录，导出文本和字幕 (不依赖大模型)")
    print("  3. 自动检测 - 检测大模型可用性，自动选择模式")
    print()
    
    choice = input("请选择 (1/2/3, 默认3): ").strip() or "3"
    
    if choice == "1":
        return "full"
    elif choice == "2":
        return "fallback"
    else:  # choice == "3" or default
        print("\n检测大模型服务可用性...")
        if check_vlm_available(config):
            print("大模型服务可用，使用完整模式")
            return "full"
        else:
            print("大模型服务不可用，切换到转录模式")
            return "fallback"


def main():
    """Main entry point."""
    print_banner()
    
    # Get video path
    video_path = get_video_path()
    print(f"选择文件: {video_path.name}")
    print()
    
    # Create default config
    config = SummarizerConfig(
        output_dir="./output",
        cache_dir="./cache"
    )
    
    # Select mode
    mode = select_mode(config)
    print()
    
    # Run pipeline
    try:
        if mode == "full":
            print("启动完整模式 (VLM 分析)")
            pipeline = VideoSummarizerPipeline(config)
            result = pipeline.summarize(video_path)
            
            print(f"\n摘要生成成功!")
            print(f"   章节数: {len(result.chapters)}")
            print(f"   输出文件: {result.output_path}")
            print(f"   处理时间: {result.processing_time:.1f}秒")
        else:
            print("启动转录模式 (仅导出文本)")
            pipeline = FallbackPipeline(config)
            result = pipeline.process(video_path)
            
            print(f"\n转录完成!")
            print(f"   语音片段数: {result.transcript_segments}")
            print(f"   输出文件:")
            for key, path in result.output_files.items():
                if key != "segment_files":
                    print(f"     - {path.name}")
            print(f"   处理时间: {result.processing_time:.1f}秒")
        
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n按 Enter 键退出...")
    input()


if __name__ == "__main__":
    main()
