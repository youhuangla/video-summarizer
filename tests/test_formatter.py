"""Tests for output formatter."""

import pytest
from pathlib import Path
from video_summarizer.output.formatter import MarkdownFormatter


def test_format_timestamp():
    """Test timestamp formatting."""
    formatter = MarkdownFormatter()
    
    assert formatter._format_timestamp(125.5) == "02:05"
    assert formatter._format_timestamp(3661) == "01:01:01"
    assert formatter._format_timestamp(45) == "00:45"
    assert formatter._format_timestamp(0) == "00:00"


def test_format_duration():
    """Test duration formatting."""
    formatter = MarkdownFormatter()
    
    assert formatter._format_duration(45) == "45秒"
    assert formatter._format_duration(125) == "2分5秒"
    assert formatter._format_duration(3665) == "1小时1分"


def test_format_basic(tmp_path):
    """Test basic Markdown formatting."""
    formatter = MarkdownFormatter()
    
    chapters = [
        {
            "start_time": 0,
            "end_time": 120,
            "title": "第一章",
            "summary": "这是第一章的摘要",
            "key_quotes": ["重要台词"],
        }
    ]
    
    result = formatter.format(
        video_path=Path("test.mp4"),
        duration=300,
        overall_summary="整体摘要",
        chapters=chapters
    )
    
    assert "# 视频摘要: test.mp4" in result
    assert "## 整体摘要" in result
    assert "## 章节概览" in result
    assert "## 章节详情" in result
    assert "第一章" in result
    assert "重要台词" in result


def test_format_save_to_file(tmp_path):
    """Test saving to file."""
    formatter = MarkdownFormatter()
    output_file = tmp_path / "summary.md"
    
    chapters = [
        {
            "start_time": 0,
            "end_time": 60,
            "title": "测试",
            "summary": "测试摘要",
            "key_quotes": [],
        }
    ]
    
    formatter.format(
        video_path=Path("test.mp4"),
        duration=60,
        overall_summary="整体",
        chapters=chapters,
        output_path=output_file
    )
    
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "测试" in content


def test_format_with_optional_fields():
    """Test formatting with optional fields."""
    formatter = MarkdownFormatter()
    
    chapters = [
        {
            "start_time": 0,
            "end_time": 100,
            "title": "完整章节",
            "summary": "摘要内容",
            "key_quotes": ["台词1", "台词2"],
            "visual_details": "画面描述",
            "topics": ["主题A", "主题B"]
        }
    ]
    
    result = formatter.format(
        video_path=Path("test.mp4"),
        duration=100,
        overall_summary="整体",
        chapters=chapters
    )
    
    assert "画面描述" in result
    assert "主题标签" in result
    assert "主题A" in result
