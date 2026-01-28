"""Output formatters for video summaries.

Generates Markdown and other formats from analysis results.
"""

from pathlib import Path
from typing import List, Dict
from datetime import timedelta


class MarkdownFormatter:
    """Format video summary as Markdown."""
    
    def format(
        self,
        video_path: Path,
        duration: float,
        overall_summary: str,
        chapters: List[Dict],
        output_path: Path = None
    ) -> str:
        """Format summary as Markdown.
        
        Args:
            video_path: Original video path
            duration: Video duration in seconds
            overall_summary: Overall video summary
            chapters: List of chapter dictionaries
            output_path: Optional output file path
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
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
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"Summary saved to: {output_path}")
        
        return markdown
    
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
