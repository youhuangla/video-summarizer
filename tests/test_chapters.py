"""Tests for chapter analyzer."""

import pytest
from video_summarizer.analyzers.chapters import ChapterAnalyzer, Chapter


def test_chapter_dataclass():
    """Test Chapter dataclass creation."""
    chapter = Chapter(
        start_time=0,
        end_time=120,
        title="测试章节",
        summary="这是一个测试摘要",
        key_quotes=["台词1", "台词2"],
        key_frames=["frame1.jpg"]
    )
    assert chapter.start_time == 0
    assert chapter.end_time == 120
    assert chapter.title == "测试章节"


def test_chapter_analyzer_init():
    """Test ChapterAnalyzer initialization."""
    analyzer = ChapterAnalyzer(api_key="test-key")
    assert analyzer.kimi is not None


def test_parse_chapter_response_valid():
    """Test parsing valid chapter response."""
    analyzer = ChapterAnalyzer(api_key="test")
    
    response = '''
    {
      "chapters": [
        {"start": 0, "end": 100, "title": "第一章", "reasoning": "开场"},
        {"start": 95, "end": 200, "title": "第二章", "reasoning": "主体"}
      ]
    }
    '''
    
    chapters = analyzer._parse_chapter_response(response, 200)
    assert len(chapters) == 2
    assert chapters[0]["title"] == "第一章"
    assert chapters[0]["start"] == 0


def test_parse_chapter_response_invalid_json():
    """Test parsing invalid JSON falls back to equal splits."""
    analyzer = ChapterAnalyzer(api_key="test")
    
    response = "Not valid JSON"
    chapters = analyzer._parse_chapter_response(response, 300)
    
    assert len(chapters) == 3
    assert chapters[0]["start"] == 0
    assert chapters[-1]["end"] == 300


def test_parse_summary_response_valid():
    """Test parsing valid summary response."""
    analyzer = ChapterAnalyzer(api_key="test")
    
    response = '''
    {
      "summary": "这是摘要",
      "key_quotes": ["台词1"],
      "topics": ["主题1"]
    }
    '''
    
    result = analyzer._parse_summary_response(response)
    assert result["summary"] == "这是摘要"
    assert result["key_quotes"] == ["台词1"]


def test_parse_summary_response_invalid():
    """Test parsing invalid summary response falls back gracefully."""
    analyzer = ChapterAnalyzer(api_key="test")
    
    response = "纯文本响应"
    result = analyzer._parse_summary_response(response)
    
    assert "summary" in result
    assert result["summary"] == "纯文本响应"
