"""Testes do schema de transcrição e do MockTranscriptionProvider."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_video_editor.transcription.exceptions import TranscriptionFailedError
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider
from ai_video_editor.transcription.schema import (
    Transcript,
    TranscriptSegment,
    TranscriptionConfig,
    Word,
)


def _segment(seg_id: int, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(id=seg_id, start_sec=start, end_sec=end, text="x")


def _transcript(segments: list[TranscriptSegment]) -> Transcript:
    return Transcript(
        source_path=Path("a.mp4"), language="pt", duration_sec=10.0, model_used="t", segments=segments
    )


def test_config_defaults():
    cfg = TranscriptionConfig()
    assert (cfg.model_size, cfg.language, cfg.device, cfg.compute_type) == ("small", None, "auto", "auto")
    assert TranscriptionConfig(language="pt").language == "pt"


def test_word_requires_start_before_end():
    with pytest.raises(ValidationError):
        Word(text="a", start_sec=1.0, end_sec=1.0)
    with pytest.raises(ValidationError):
        Word(text="a", start_sec=2.0, end_sec=1.0)


def test_segment_requires_start_before_end():
    with pytest.raises(ValidationError):
        _segment(0, 2.0, 1.0)


def test_segment_words_must_be_chronological():
    words = [Word(text="b", start_sec=1.0, end_sec=1.5), Word(text="a", start_sec=0.0, end_sec=0.5)]
    with pytest.raises(ValidationError):
        TranscriptSegment(id=0, start_sec=0.0, end_sec=2.0, text="b a", words=words)


def test_transcript_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        _transcript([_segment(0, 0.0, 1.0), _segment(0, 2.0, 3.0)])


def test_transcript_rejects_out_of_order_segments():
    with pytest.raises(ValidationError):
        _transcript([_segment(0, 5.0, 6.0), _segment(1, 1.0, 2.0)])


def test_save_and_load_roundtrip(tmp_path: Path):
    original = MockTranscriptionProvider().transcribe(Path("v.mp4"), TranscriptionConfig())
    path = original.save(tmp_path / "sub" / "transcript.json")
    assert path.is_file()
    assert Transcript.load(path) == original


def test_load_rejects_invalid_json_content(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"source_path": "a.mp4"}', encoding="utf-8")
    with pytest.raises(ValidationError):
        Transcript.load(path)


def test_mock_is_deterministic_with_sequential_ids():
    provider = MockTranscriptionProvider()
    first = provider.transcribe(Path("v.mp4"), TranscriptionConfig())
    second = provider.transcribe(Path("v.mp4"), TranscriptionConfig())
    assert first == second
    assert [s.id for s in first.segments] == [0, 1, 2]
    starts = [s.start_sec for s in first.segments]
    assert starts == sorted(starts)
    assert first.model_used == "mock"
    assert all(s.words for s in first.segments)


def test_mock_custom_sentences_and_language():
    provider = MockTranscriptionProvider(sentences=["um dois", "três"], word_duration_sec=1.0, gap_sec=2.0)
    result = provider.transcribe(Path("v.mp4"), TranscriptionConfig(language="en"))
    assert result.language == "en"
    assert [s.text for s in result.segments] == ["um dois", "três"]
    assert result.segments[0].end_sec == 2.0
    assert result.segments[1].start_sec == 4.0


def test_mock_custom_transcript_is_returned_as_is():
    custom = _transcript([_segment(0, 0.0, 1.0)])
    assert MockTranscriptionProvider(custom_transcript=custom).transcribe(
        Path("v.mp4"), TranscriptionConfig()
    ) is custom


def test_mock_forced_failure():
    with pytest.raises(TranscriptionFailedError):
        MockTranscriptionProvider(force_failure=True).transcribe(Path("v.mp4"), TranscriptionConfig())
