"""Output formatters for video summaries.

Generates Markdown and other formats from analysis results.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import timedelta, datetime


class MarkdownFormatter:
    """Format video summary as Markdown."""
    
    def __init__(
        self,
        output_timestamp: bool = True,
        output_model_in_filename: bool = True,
        output_keep_latest: bool = True,
        output_time_format: str = "%Y%m%d_%H%M%S"
    ):
        """Initialize formatter.
        
        Args:
            output_timestamp: Add timestamp suffix to filename
            output_model_in_filename: Include model short name in filename
            output_keep_latest: Maintain a "latest" copy without timestamp
            output_time_format: Timestamp format string
        """
        self.output_timestamp = output_timestamp
        self.output_model_in_filename = output_model_in_filename
        self.output_keep_latest = output_keep_latest
        self.output_time_format = output_time_format
    
    def format(
        self,
        video_path: Path,
        duration: float,
        overall_summary: str,
        chapters: List[Dict],
        output_path: Path = None,
        output_dir: Path = None,
        metadata: dict = None
    ) -> str:
        """Format summary as Markdown.
        
        Args:
            video_path: Original video path
            duration: Video duration in seconds
            overall_summary: Overall video summary
            chapters: List of chapter dictionaries
            output_path: Optional explicit output file path
            output_dir: Optional output directory (uses current dir if None)
            metadata: Optional metadata dict with keys:
                     - timestamp: str (formatted timestamp)
                     - model_short: str (short model name)
                     - model_full: str (full model name)
                     
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Add YAML frontmatter with metadata (for CI/CD parsing)
        if metadata:
            lines.append("---")
            if 'timestamp' in metadata:
                lines.append(f"generated_at: {metadata['timestamp']}")
            if 'model_full' in metadata:
                lines.append(f"model: {metadata['model_full']}")
            if 'model_short' in metadata:
                lines.append(f"model_short: {metadata['model_short']}")
            lines.append(f"video_duration: {duration:.1f}")
            lines.append(f"chapters_count: {len(chapters)}")
            lines.append("---")
            lines.append("")
        
        # Header
        lines.append(f"# 视频摘要: {video_path.name}")
        lines.append("")
        lines.append(f"- **文件**: `{video_path}`")
        lines.append(f"- **时长**: {self._format_duration(duration)}")
        lines.append(f"- **章节数**: {len(chapters)}")
        lines.append("")
        
        # Overall summary
        lines.append("## 整体摘要")
        lines.append("")
        lines.append(overall_summary)
        lines.append("")
        
        # Table of contents
        lines.append("## 章节概览")
        lines.append("")
        lines.append("| 时间 | 章节 | 时长 |")
        lines.append("|------|------|------|")
        for ch in chapters:
            start = self._format_timestamp(ch['start_time'])
            duration_str = self._format_duration(ch['end_time'] - ch['start_time'])
            lines.append(f"| {start} | {ch['title']} | {duration_str} |")
        lines.append("")
        
        # Detailed chapters
        lines.append("## 章节详情")
        lines.append("")
        
        for i, ch in enumerate(chapters, 1):
            start = self._format_timestamp(ch['start_time'])
            end = self._format_timestamp(ch['end_time'])
            
            lines.append(f"### [{start} - {end}] {ch['title']}")
            lines.append("")
            
            # Summary
            lines.append("**摘要:**")
            lines.append(ch['summary'])
            lines.append("")
            
            # Key quotes
            if ch.get('key_quotes'):
                lines.append("**关键台词:**")
                for quote in ch['key_quotes']:
                    lines.append(f"> {quote}")
                lines.append("")
            
            # Visual details (if available)
            if ch.get('visual_details'):
                lines.append("**视觉细节:**")
                lines.append(ch['visual_details'])
                lines.append("")
            
            # Topics (if available)
            if ch.get('topics'):
                lines.append(f"**主题标签:** {', '.join(ch['topics'])}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        markdown = "\n".join(lines)
        
        # Save to file(s) if output directory provided
        if output_dir or output_path:
            self._save_files(
                markdown=markdown,
                video_path=video_path,
                output_dir=output_dir,
                output_path=output_path,
                metadata=metadata
            )
        
        return markdown
    
    def _save_files(
        self,
        markdown: str,
        video_path: Path,
        output_dir: Path = None,
        output_path: Path = None,
        metadata: dict = None
    ):
        """Save markdown to file(s) with optional timestamp and latest copy."""
        # Determine output directory
        if output_path:
            output_dir = Path(output_path).parent
        elif output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path("./output")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = video_path.stem
        
        # Build filename components
        parts = [video_name]
        
        if self.output_model_in_filename and metadata and metadata.get('model_short'):
            parts.append(metadata['model_short'])
        
        if self.output_timestamp and metadata and metadata.get('timestamp'):
            parts.append(metadata['timestamp'])
        
        # Generate timestamped filename
        if len(parts) > 1:
            timestamped_name = "_".join(parts) + "_summary.md"
        else:
            timestamped_name = f"{video_name}_summary.md"
        
        timestamped_path = output_dir / timestamped_name
        timestamped_path.write_text(markdown, encoding="utf-8")
        print(f"Summary saved to: {timestamped_path.name.encode('ascii', 'ignore').decode()}")
        
        # Create "latest" copy if enabled and we have a timestamped version
        if self.output_keep_latest and len(parts) > 1:
            latest_name = f"{video_name}_summary.md"
            latest_path = output_dir / latest_name
            shutil.copy(timestamped_path, latest_path)
            print(f"Latest copy saved to: {latest_path.name.encode('ascii', 'ignore').decode()}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable form."""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds/60)}分{int(seconds%60)}秒"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}小时{mins}分"
