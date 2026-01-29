"""Fallback pipeline for when VLM services are unavailable.

This pipeline only performs:
1. Audio transcription (Whisper)
2. Export transcripts in multiple formats (TXT, JSON, SRT, segments)

No VLM calls are made in this mode.
"""

import time
from pathlib import Path
from typing import Union, Dict
from dataclasses import dataclass

from video_summarizer.config import SummarizerConfig
from video_summarizer.extractors.metadata import MetadataExtractor
from video_summarizer.extractors.audio import AudioExtractor, TranscriptSegment
from video_summarizer.output.transcript_exporter import TranscriptExporter


@dataclass
class FallbackResult:
    """Result from fallback pipeline."""
    video_path: Path
    duration: float
    transcript_segments: int
    output_files: Dict[str, Path]
    processing_time: float


class FallbackPipeline:
    """Pipeline that only extracts and exports transcripts (no VLM)."""
    
    def __init__(self, config: SummarizerConfig = None):
        """Initialize fallback pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config or SummarizerConfig()
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        self.audio_extractor = AudioExtractor(
            api_base=self.config.whisper_base_url,
            api_key=self.config.whisper_api_key,
            cache_dir=self.config.cache_dir,
            model=self.config.whisper_model
        )
        self.exporter = TranscriptExporter(output_dir=self.config.output_dir)
    
    def process(
        self,
        video_path: Union[str, Path]
    ) -> FallbackResult:
        """Process video - extract and export transcripts only.
        
        Args:
            video_path: Path to video file
            
        Returns:
            FallbackResult with exported file paths
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        print("\n" + "=" * 60)
        print("Video Transcript Extractor (Fallback Mode)")
        print("=" * 60)
        print(f"File: {video_path.name.encode('ascii', 'ignore').decode()}")
        print()
        
        # Step 1: Extract metadata
        print("[1/3] Extracting video metadata...")
        metadata = self.metadata_extractor.extract(video_path)
        print(f"      Duration: {metadata.duration:.1f}s, Resolution: {metadata.width}x{metadata.height}")
        print()
        
        # Step 2: Transcribe audio
        print("[2/3] Transcribing audio...")
        transcript_segments = self.audio_extractor.extract_transcript(video_path)
        print(f"      Found {len(transcript_segments)} speech segments")
        
        if not transcript_segments:
            print("      Warning: No speech detected in video")
        print()
        
        # Step 3: Export in all formats
        print("[3/3] Exporting transcripts...")
        output_files = self.exporter.export_all(
            video_path=video_path,
            segments=transcript_segments,
            video_duration=metadata.duration
        )
        
        # Print output summary
        print(f"      Full TXT: {output_files['full_txt'].name.encode('ascii', 'ignore').decode()}")
        print(f"      Full JSON: {output_files['full_json'].name.encode('ascii', 'ignore').decode()}")
        print(f"      Segments: {len(output_files['segment_files'])} files")
        print(f"      SRT: {output_files['srt'].name.encode('ascii', 'ignore').decode()}")
        print(f"      Metadata: {output_files['metadata'].name.encode('ascii', 'ignore').decode()}")
        print()
        
        # Done
        processing_time = time.time() - start_time
        print(f"Done! Total time: {processing_time:.1f}s")
        print("=" * 60)
        print()
        
        return FallbackResult(
            video_path=video_path,
            duration=metadata.duration,
            transcript_segments=len(transcript_segments),
            output_files=output_files,
            processing_time=processing_time
        )


def check_vlm_available(config: SummarizerConfig) -> bool:
    """Check if VLM service is available.
    
    Args:
        config: Configuration with VLM settings
        
    Returns:
        True if VLM is available, False otherwise
    """
    import requests
    
    try:
        # Simple health check - try to list models
        response = requests.get(
            f"{config.base_url}/models",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False
