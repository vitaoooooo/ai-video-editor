"""Testes do WER normalizado, da cauda após o fim da referência, das transcrições salvas e do rescore."""

import json
from pathlib import Path

import pytest

from ai_video_editor.benchmark import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkRun,
    RescoreError,
    compute_wer,
    compute_wer_words,
    format_markdown,
    normalize_words,
    rescore_run,
    run_benchmark,
    save_run,
)
from ai_video_editor.benchmark.runner import score_texts
from ai_video_editor.cli import main
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider


# --- normalizador (en) -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("one two three", ["1", "2", "3"]),  # números seguidos continuam separados
        ("zero", ["0"]),
        ("seventeen", ["17"]),
        ("twenty one", ["21"]),
        ("twenty-one", ["21"]),
        ("ninety nine", ["99"]),
        ("a hundred", ["100"]),
        ("one hundred", ["100"]),
        ("two hundred and five", ["205"]),
        ("three hundred forty two", ["342"]),
        ("a hundred and twenty", ["120"]),
        ("two thousand five", ["2005"]),
        ("a thousand", ["1000"]),
        ("1,000 apples", ["1000", "apples"]),
        ("42 and 7", ["42", "and", "7"]),
        ("a cat", ["a", "cat"]),
        ("and twenty", ["and", "20"]),
    ],
)
def test_english_numbers(text, expected):
    assert normalize_words(text, "en") == expected


def test_english_spelling_variants():
    assert normalize_words("Okay, OK.", "en") == ["ok", "ok"]
    assert normalize_words("the x's and y’s", "en") == ["the", "xs", "and", "ys"]
    assert normalize_words("don't", "en") == ["don't"]  # outras contrações ficam
    assert normalize_words("right-hand side", "en") == ["right", "hand", "side"]


def test_unknown_language_only_basic_normalization():
    assert normalize_words("Okay, two", "xx") == ["okay", "two"]
    assert normalize_words("Okay, two", None) == ["okay", "two"]


def test_normalized_wer_ignores_spelling_but_raw_does_not():
    reference = "Okay so x's are one and two, twenty-one in all"
    hypothesis = "OK so xs are 1 and 2, 21 in all"
    raw, normalized = score_texts(reference, hypothesis, "en")
    assert raw.errors > 0
    assert normalized is not None and normalized.wer == 0.0
    assert normalized.reference_words == normalized.hypothesis_words == 10
    assert compute_wer(reference, hypothesis).substitutions == raw.substitutions


def test_normalized_is_none_without_language_rules():
    raw, normalized = score_texts("one two", "1 2", "pt")
    assert normalized is None and raw.substitutions == 2


# --- cauda depois do fim da referência ---------------------------------------------------


def test_tail_after_reference_is_not_counted_as_insertions():
    reference = "the quick brown fox jumps"
    hypothesis = "the quick brown fox jumps over the lazy dog and keeps running far away"
    result = compute_wer(reference, hypothesis)
    assert result.tail_words == 9
    assert result.insertions == 0 and result.wer == 0.0
    assert result.tail_preview.startswith("over the lazy dog")
    assert result.hypothesis_words == 14


def test_tail_does_not_hide_middle_insertions_or_errors():
    reference = "a b c d e f"
    hypothesis = "a b x c d e g h i j k l m"  # x no meio, "f" trocado e cauda
    result = compute_wer(reference, hypothesis)
    assert result.insertions == 1  # só o x
    assert result.substitutions == 1  # f → g
    assert result.tail_words == 6  # g h i j k l m, menos a vaga de "f" que o "g" ocupa
    assert result.hypothesis_words == 13


def test_short_trailing_words_stay_insertions():
    result = compute_wer("a b c d", "a b c d e f")
    assert result.tail_words == 0 and result.insertions == 2


def test_no_tail_when_hypothesis_ends_like_reference():
    result = compute_wer("a b c d e", "x y z w v a b c d e")
    assert result.tail_words == 0 and result.insertions == 5


def test_tail_pooled_in_average_and_shown_in_report(tmp_path):
    run = _run_mock(tmp_path, ["um dois três quatro cinco seis sete oito nove dez onze doze"], "um dois três quatro cinco")
    config_run = run.configs[0]
    assert config_run.items[0].wer.tail_words == 7
    assert run.averages[0].wer.tail_words == 7
    markdown = format_markdown(run)
    assert "Após a ref." in markdown and "## Palavras após o fim da referência" in markdown


# --- transcrições salvas e rescore -------------------------------------------------------


def _write_audio(path: Path) -> Path:
    path.write_bytes(b"audio")
    return path


def _run_mock(tmp_path: Path, sentences, reference_text, language="en", out_name="run"):
    reference = tmp_path / "ref.txt"
    reference.write_text("# fonte\n" + reference_text + "\n", encoding="utf-8")
    item = BenchmarkItem(
        name="aula", audio=_write_audio(tmp_path / "a.wav"), reference=reference, language=language
    )
    out_dir = tmp_path / out_name
    out_dir.mkdir(exist_ok=True)
    run = run_benchmark(
        "t",
        [BenchmarkConfig.parse("mock/m:cpu:int8")],
        [item],
        provider_factories={"mock": lambda config: MockTranscriptionProvider(sentences=sentences)},
        out_dir=out_dir,
    )
    return run


def test_hypotheses_are_saved_per_config(tmp_path):
    run = _run_mock(tmp_path, ["hello world"], "hello world")
    result = run.configs[0].items[0]
    assert result.hypothesis_file == "hypotheses/mock_m_cpu_int8.txt"
    assert (tmp_path / "run" / result.hypothesis_file).read_text(encoding="utf-8").strip() == "hello world"


def test_hypotheses_path_with_several_items(tmp_path):
    items = [
        BenchmarkItem(name="um", audio=_write_audio(tmp_path / "a.wav")),
        BenchmarkItem(name="dois", audio=_write_audio(tmp_path / "b.wav")),
    ]
    run = run_benchmark(
        "t",
        [BenchmarkConfig.parse("mock/m:cpu:int8")],
        items,
        provider_factories={"mock": lambda config: MockTranscriptionProvider(sentences=["oi"])},
        out_dir=tmp_path,
    )
    assert [r.hypothesis_file for r in run.configs[0].items] == [
        "hypotheses/um/mock_m_cpu_int8.txt",
        "hypotheses/dois/mock_m_cpu_int8.txt",
    ]
    assert (tmp_path / "hypotheses" / "dois" / "mock_m_cpu_int8.txt").is_file()


def test_rescore_from_fake_saved_transcripts(tmp_path):
    reference = tmp_path / "saved_ref.txt"
    reference.write_text("# c\nOkay so one and two and three\n", encoding="utf-8")
    run_dir = tmp_path / "exec"
    (run_dir / "hypotheses").mkdir(parents=True)
    (run_dir / "hypotheses" / "mock_m_cpu_int8.txt").write_text("OK so 1 and 2 and 3\n", encoding="utf-8")
    first = _run_mock(tmp_path, ["so"], "so", out_name="orig")  # só para obter um results.json válido
    data = json.loads(first.model_dump_json())
    data["items"][0]["reference"] = str(reference)
    data["items"][0]["language"] = "en"
    result = data["configs"][0]["items"][0]
    result.update(hypothesis_file="hypotheses/mock_m_cpu_int8.txt", wer=None, wer_normalized=None)
    data["averages"] = []
    (run_dir / "results.json").write_text(json.dumps(data), encoding="utf-8")

    rescored = rescore_run(run_dir)
    scored = rescored.configs[0].items[0]
    assert scored.wer.errors == 4 and scored.wer_normalized.wer == 0.0  # okay→ok, 1, 2, 3 só no bruto
    assert rescored.averages[0].wer_normalized.wer == 0.0
    # o tempo medido na execução original é preservado
    assert scored.transcription_time_sec == first.configs[0].items[0].transcription_time_sec

    markdown = save_run(rescored, run_dir)
    assert "WER norm." in markdown
    assert BenchmarkRun.model_validate_json((run_dir / "results.json").read_text(encoding="utf-8"))
    assert (run_dir / "results.md").read_text(encoding="utf-8").startswith("# Benchmark")


def test_rescore_without_saved_hypotheses_fails_clearly(tmp_path):
    run = _run_mock(tmp_path, ["a b"], "a b")
    run.configs[0].items[0].hypothesis_file = None
    run_dir = tmp_path / "old"
    save_run(run, run_dir)
    with pytest.raises(RescoreError, match="rode o benchmark de novo"):
        rescore_run(run_dir)


def test_rescore_cli(tmp_path, monkeypatch, capsys):
    _run_mock(tmp_path, ["hello there"], "hello here")
    # results.json da execução real do mock, com a referência em caminho absoluto
    run_dir = tmp_path / "run"
    run = _run_mock(tmp_path, ["hello there"], "hello here")
    save_run(run, run_dir)
    monkeypatch.setattr("sys.argv", ["x", "benchmark", "rescore", str(run_dir)])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert "WER bruto" in capsys.readouterr().out
    assert compute_wer_words(["a"], ["a"]).wer == 0.0
