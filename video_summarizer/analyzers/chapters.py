"""Chapter analysis using Kimi VLM.

Implements the chapter boundary detection and summary generation
using pyramidal perception approach inspired by Video-Browser.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

from video_summarizer.utils.kimi_client import KimiClient


@dataclass
class Chapter:
    """A video chapter with summary."""
    start_time: float
    end_time: float
    title: str
    summary: str
    key_quotes: List[str]
    key_frames: List[str]  # Frame paths


class ChapterAnalyzer:
    """Analyze video and generate chapter summaries."""
    
    def __init__(self, api_key: str, base_url: str = "http://127.0.0.1:1234/v1", model: str = None):
        """Initialize analyzer.
        
        Args:
            api_key: API key (can be "EMPTY" for local APIs)
            base_url: API base URL
            model: Model name (optional)
        """
        self.kimi = KimiClient(api_key=api_key, base_url=base_url, model=model)
    
    def detect_chapters(
        self,
        frames: List,
        transcript_text: str,
        duration: float,
        min_chapter_duration: int = 30,
        max_chapters: int = 8
    ) -> List[Dict]:
        """Detect chapter boundaries from sparse frames.
        
        Args:
            frames: List of FrameInfo objects (sparse sampling)
            transcript_text: Full transcript text
            duration: Video duration in seconds
            min_chapter_duration: Minimum chapter length
            max_chapters: Maximum number of chapters
            
        Returns:
            List of chapter boundaries with start/end times
        """
        # Build frame descriptions
        frame_timestamps = [f.timestamp for f in frames]
        
        prompt = f"""你是一个专业的视频编辑助手。请分析以下视频的关键帧和字幕，识别出自然的章节边界。

视频信息：
- 总时长: {duration:.0f}秒
- 关键帧时间点: {frame_timestamps}

字幕节选（前3000字符）：
{transcript_text[:3000]}

任务：
1. 识别 {min(3, int(duration/60))}-{min(max_chapters, max(3, int(duration/120)))} 个主要章节
2. 每个章节必须包含：开始时间(秒)、结束时间(秒)、章节标题(5-15字中文)
3. 章节边界应对应内容主题的明显转换（话题切换、场景切换、引入新概念等）
4. 每个章节至少 {min_chapter_duration} 秒
5. 第一个章节从 0 开始，最后一个章节到 {duration:.0f} 结束

重要：返回严格的JSON格式，不要其他内容。

输出格式：
{{
  "chapters": [
    {{
      "start": 0,
      "end": 120,
      "title": "开场与背景介绍",
      "reasoning": "视频开始介绍背景和主题..."
    }},
    {{
      "start": 115,
      "end": 300,
      "title": "核心内容演示",
      "reasoning": "进入主要内容展示..."
    }}
  ]
}}
"""
        
        # Get frame paths
        frame_paths = [f.path for f in frames]
        
        # Call Kimi VLM
        response = self.kimi.analyze_images(
            images=frame_paths,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.3
        )
        
        # Parse JSON response
        return self._parse_chapter_response(response, duration)
    
    def summarize_chapter(
        self,
        frames: List,
        transcript_text: str,
        title_hint: str = "",
        detail_level: str = "standard"
    ) -> Dict:
        """Generate detailed summary for a chapter.
        
        Args:
            frames: Dense frame sampling within chapter
            transcript_text: Transcript text for this chapter
            title_hint: Suggested chapter title
            detail_level: "brief", "standard", or "detailed"
            
        Returns:
            Dictionary with summary, key quotes, etc.
        """
        # Determine output fields based on detail level
        if detail_level == "brief":
            fields = "summary (2-3句话)"
        elif detail_level == "standard":
            fields = "summary (3-5句话), key_quotes (2-3句关键台词)"
        else:  # detailed
            fields = "summary (详细段落), key_quotes (3-5句), visual_details (视觉细节描述), topics (3-5个关键词)"
        
        prompt = f"""基于以下视频帧和字幕，为"{title_hint}"这个章节生成详细摘要。

字幕内容：
{transcript_text[:4000]}

任务：
1. 概括这个章节的核心内容
2. 提取最代表性的台词/引语
3. 描述关键视觉信息（如果有重要画面）

输出格式（JSON）：
{{
  "summary": "章节摘要...",
  "key_quotes": ["台词1", "台词2"],
  "visual_details": "画面描述..." (可选),
  "topics": ["主题1", "主题2"] (可选)
}}

要求：摘要必须是中文，简洁准确。
"""
        
        frame_paths = [f.path for f in frames[:6]]  # Limit frames
        
        response = self.kimi.analyze_images(
            images=frame_paths,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.4
        )
        
        return self._parse_summary_response(response)
    
    def _parse_chapter_response(self, response: str, duration: float) -> List[Dict]:
        """Parse chapter detection response."""
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                chapters = data.get("chapters", [])
                
                # Validate and fix boundaries
                validated = []
                for i, ch in enumerate(chapters):
                    start = max(0, float(ch.get("start", 0)))
                    end = min(duration, float(ch.get("end", duration)))
                    
                    # Ensure reasonable duration
                    if end - start < 30:
                        if i == len(chapters) - 1:
                            start = max(0, end - 30)
                        else:
                            end = min(duration, start + 30)
                    
                    validated.append({
                        "start": start,
                        "end": end,
                        "title": ch.get("title", f"章节 {i+1}"),
                        "reasoning": ch.get("reasoning", "")
                    })
                
                return validated
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse chapter response: {e}")
        
        # Fallback: split into 3 equal parts
        segment = duration / 3
        return [
            {"start": 0, "end": segment, "title": "第一部分", "reasoning": "自动分割"},
            {"start": segment * 0.9, "end": segment * 2, "title": "第二部分", "reasoning": "自动分割"},
            {"start": segment * 1.9, "end": duration, "title": "第三部分", "reasoning": "自动分割"}
        ]
    
    def _parse_summary_response(self, response: str) -> Dict:
        """Parse summary response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
        
        # Fallback: treat entire response as summary
        return {
            "summary": response[:500],
            "key_quotes": [],
            "visual_details": "",
            "topics": []
        }
