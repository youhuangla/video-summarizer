"""Main video summarization pipeline.

Orchestrates the entire summarization process:
1. Extract metadata
2. Transcribe audio (via local HTTP API)
3. Extract frames (pyramidal sampling)
4. Detect chapters
5. Summarize each chapter
6. Generate output
"""

import time
from pathlib import Path
from typing import Union, List, Dict
from dataclasses import dataclass

from video_summarizer.config import SummarizerConfig
from video_summarizer.extractors.metadata import MetadataExtractor
from video_summarizer.extractors.audio import AudioExtractor
from video_summarizer.extractors.frames import FrameExtractor
from video_summarizer.analyzers.chapters import ChapterAnalyzer
from video_summarizer.output.formatter import MarkdownFormatter


@dataclass
class VideoSummaryResult:
    """Result container for video summarization."""
    video_path: Path
    duration: float
    overall_summary: str
    chapters: List[Dict]
    processing_time: float
    output_path: Path = None


class VideoSummarizerPipeline:
    """Main pipeline for video summarization."""
    
    def __init__(self, config: SummarizerConfig = None):
        """Initialize pipeline.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or SummarizerConfig()
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        self.audio_extractor = AudioExtractor(
            api_base="http://127.0.0.1:18181/v1/audio",
            cache_dir=self.config.cache_dir
        )
        self.frame_extractor = FrameExtractor(
            cache_dir=f"{self.config.cache_dir}/frames"
        )
        self.chapter_analyzer = ChapterAnalyzer(
            api_key=self.config.kimi_api_key,
            base_url=self.config.kimi_base_url
        )
        self.formatter = MarkdownFormatter()
    
    def summarize(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path] = None
    ) -> VideoSummaryResult:
        """Summarize a video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional output markdown file path
            
        Returns:
            VideoSummaryResult object
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        print(f"\n{'='*60}")
        print(f"视频摘要生成")
        print(f"{'='*60}")
        print(f"文件: {video_path}")
        print()
        
        # Step 1: Extract metadata
        print("[1/6] 提取视频元信息...")
        metadata = self.metadata_extractor.extract(video_path)
        print(f"      时长: {metadata.duration:.1f}s, 分辨率: {metadata.width}x{metadata.height}")
        print()
        
        # Step 2: Transcribe audio
        print("[2/6] 提取音频并转录...")
        try:
            transcript_segments = self.audio_extractor.extract_transcript(video_path)
            full_transcript = " ".join([s.text for s in transcript_segments])
            print(f"      识别到 {len(transcript_segments)} 个语音片段")
        except ConnectionError as e:
            print(f"      警告: 无法连接 Whisper 服务 ({e})")
            print("      将继续使用空字幕...")
            transcript_segments = []
            full_transcript = ""
        print()
        
        # Step 3: Process video
        if metadata.duration > self.config.segment_duration:
            print(f"[3/6] 视频较长，分段处理...")
            chapters = self._process_long_video(
                video_path, transcript_segments, metadata.duration
            )
        else:
            print(f"[3/6] 分析视频章节...")
            chapters = self._process_segment(
                video_path, transcript_segments, 0, metadata.duration
            )
        
        print(f"      检测到 {len(chapters)} 个章节")
        print()
        
        # Step 4: Generate overall summary
        print("[4/6] 生成整体摘要...")
        overall_summary = self._generate_overall_summary(chapters, full_transcript)
        print(f"      {overall_summary[:100]}...")
        print()
        
        # Step 5: Format output
        print("[5/6] 格式化输出...")
        if output_path is None:
            output_path = Path(self.config.output_dir) / f"{video_path.stem}_summary.md"
        else:
            output_path = Path(output_path)
        
        self.formatter.format(
            video_path=video_path,
            duration=metadata.duration,
            overall_summary=overall_summary,
            chapters=chapters,
            output_path=output_path
        )
        print()
        
        # Step 6: Done
        processing_time = time.time() - start_time
        print(f"[6/6] 完成! 总耗时: {processing_time:.1f}秒")
        print(f"{'='*60}\n")
        
        return VideoSummaryResult(
            video_path=video_path,
            duration=metadata.duration,
            overall_summary=overall_summary,
            chapters=chapters,
            processing_time=processing_time,
            output_path=output_path
        )
    
    def _process_segment(
        self,
        video_path: Path,
        transcript_segments: List,
        start_offset: float,
        end_offset: float
    ) -> List[Dict]:
        """Process a video segment."""
        duration = end_offset - start_offset
        
        # Get transcript for this segment
        segment_transcript = " ".join([
            s.text for s in transcript_segments
            if s.start >= start_offset and s.end <= end_offset
        ])
        
        # Stage 1: Sparse sampling for chapter detection
        sparse_frames = self.frame_extractor.extract_uniform(
            video_path,
            num_frames=self.config.sparse_frame_count,
            start_time=start_offset,
            end_time=end_offset
        )
        
        # Detect chapters
        chapter_boundaries = self.chapter_analyzer.detect_chapters(
            frames=sparse_frames,
            transcript_text=segment_transcript,
            duration=duration,
            min_chapter_duration=self.config.min_chapter_duration,
            max_chapters=self.config.max_chapters
        )
        
        # Stage 2: Dense sampling and summarization
        chapters = []
        for boundary in chapter_boundaries:
            # Adjust times for segment offset
            ch_start = start_offset + boundary['start']
            ch_end = start_offset + boundary['end']
            
            # Extract dense frames
            dense_frames = self.frame_extractor.extract_range(
                video_path,
                start_time=ch_start,
                end_time=ch_end,
                fps=self.config.dense_fps
            )
            
            # Get chapter transcript
            ch_transcript = " ".join([
                s.text for s in transcript_segments
                if s.start >= ch_start and s.end <= ch_end
            ])
            
            # Summarize chapter
            summary = self.chapter_analyzer.summarize_chapter(
                frames=dense_frames,
                transcript_text=ch_transcript,
                title_hint=boundary['title'],
                detail_level="standard"
            )
            
            chapters.append({
                "start_time": ch_start,
                "end_time": ch_end,
                "title": boundary['title'],
                **summary
            })
        
        return chapters
    
    def _process_long_video(
        self,
        video_path: Path,
        transcript_segments: List,
        duration: float
    ) -> List[Dict]:
        """Process long video by splitting into segments."""
        segment_length = self.config.segment_duration
        overlap = self.config.overlap_duration
        
        all_chapters = []
        segment_starts = list(range(0, int(duration), segment_length - overlap))
        
        for i, seg_start in enumerate(segment_starts):
            seg_end = min(seg_start + segment_length, duration)
            print(f"      处理分段 {i+1}/{len(segment_starts)}: {seg_start:.0f}s-{seg_end:.0f}s")
            
            chapters = self._process_segment(
                video_path, transcript_segments, seg_start, seg_end
            )
            
            # Merge overlapping chapters
            if all_chapters and chapters:
                last_end = all_chapters[-1]['end_time']
                chapters = [c for c in chapters if c['start_time'] >= last_end - overlap]
            
            all_chapters.extend(chapters)
            
            if seg_end >= duration:
                break
        
        return self._merge_short_chapters(all_chapters)
    
    def _merge_short_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """Merge chapters that are too short."""
        if not chapters:
            return chapters
        
        merged = [chapters[0]]
        
        for ch in chapters[1:]:
            duration = ch['end_time'] - ch['start_time']
            if duration < self.config.min_chapter_duration:
                # Merge with previous
                merged[-1]['end_time'] = ch['end_time']
                merged[-1]['summary'] += f" {ch['summary']}"
            else:
                merged.append(ch)
        
        return merged
    
    def _generate_overall_summary(self, chapters: List[Dict], transcript: str) -> str:
        """Generate overall video summary."""
        # Build chapter summaries
        chapter_text = "\n".join([
            f"章节 {i+1} ({self._fmt_time(c['start_time'])}-{self._fmt_time(c['end_time'])}): {c['title']}\n{c['summary'][:150]}..."
            for i, c in enumerate(chapters)
        ])
        
        prompt = f"""基于以下视频章节摘要，生成一个简洁的整体摘要（100字以内）：

{chapter_text}

请用一句话概括视频主题，然后列出3个核心要点。只输出摘要文本，不要JSON格式。
"""
        
        return self.chapter_analyzer.kimi.generate_text(prompt, max_tokens=300)
    
    def _fmt_time(self, seconds: float) -> str:
        """Format time as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
