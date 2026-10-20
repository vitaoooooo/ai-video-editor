"""Testes do benchmark de transcrição: WER, normalização, manifesto, tabela e execução com mock."""

import json
from pathlib import Path

import pytest

from ai_video_editor.benchmark import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkManifest,
    BenchmarkRun,
    compute_wer,
    format_markdown,
    normalize_text,
    run_benchmark,
)
from ai_video_editor.benchmark.runner import load_reference
from ai_video_editor.cli import build_parser, main
from ai_video_editor.transcription.providers.mock import MockTranscriptionProvider


def test_wer_identical_is_zero():
    result = compute_wer("the quick brown fox", "the quick brown fox")
    assert result.wer == 0.0
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 0)
    assert result.reference_words == 4


def test_wer_one_substitution():
    result = compute_wer("the quick brown fox", "the quick brown box")
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.wer == pytest.approx(0.25)
    assert [(s.reference, s.hypothesis) for s in result.top_substitutions] == [("fox", "box")]


def test_wer_one_deletion():
    result = compute_wer("the quick brown fox", "the brown fox")
    assert (result.substitutions, result.deletions, result.insertions) == (0, 1, 0)
    assert result.wer == pytest.approx(0.25)


def test_wer_one_insertion():
    result = compute_wer("the quick fox", "the quick brown fox")
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 1)
    assert result.wer == pytest.approx(1 / 3)
    assert result.hypothesis_words == 4


def test_wer_mixed_and_top_substitutions_counted():
    result = compute_wer("a gnu b gnu c d", "a canoe b canoe c")
    assert result.substitutions == 2
    assert result.deletions == 1
    assert result.top_substitutions[0].count == 2
    assert (result.top_substitutions[0].reference, result.top_substitutions[0].hypothesis) == ("gnu", "canoe")


def test_wer_empty_reference():
    assert compute_wer("", "").wer == 0.0
    assert compute_wer("", "algo").wer == 1.0


def test_normalization():
    assert normalize_text("Hello,  WORLD!") == ["hello", "world"]
    assert normalize_text("Don’t stop — right-hand side...") == ["don't", "stop", "right", "hand", "side"]
    assert normalize_text("‘quoted’ `x´") == ["quoted", "x"]
    assert compute_wer("Don't  stop.", "dont stop").substitutions == 1
    assert compute_wer("Don't stop.", "don’t STOP").wer == 0.0


def test_load_reference_ignores_comments(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("# fonte\nolá mundo\n# outro\nfim\n", encoding="utf-8")
    assert load_reference(ref) == "olá mundo\nfim"


def test_config_parse():
    config = BenchmarkConfig.parse("large-v3-turbo:cuda:float16")
    assert (config.model, config.device, config.compute_type) == ("large-v3-turbo", "cuda", "float16")
    assert config.label == "large-v3-turbo:cuda:float16"
    assert BenchmarkConfig.parse("mock/x:cpu:int8").provider == "mock"
    for bad in ("small", "small:gpu:int8", "small:cpu:"):
        with pytest.raises(ValueError):
            BenchmarkConfig.parse(bad)


def _write_audio(path: Path) -> Path:
    path.write_bytes(b"audio")
    return path


def _mock_factories(sentences, fail_models=()):
    def factory(config):
        return MockTranscriptionProvider(sentences=sentences, force_failure=config.model in fail_models)

    return {"mock": factory}


def test_run_with_mock_and_report(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("# c\nolá pessoal bem vindo\nhoje vamos editar\n", encoding="utf-8")
    items = [
        BenchmarkItem(name="com-ref", audio=_write_audio(tmp_path / "a.wav"), reference=ref),
        BenchmarkItem(name="sem-ref", audio=_write_audio(tmp_path / "b.wav")),
    ]
    configs = [BenchmarkConfig.parse("mock/ok:cpu:int8"), BenchmarkConfig.parse("mock/bad:cpu:int8")]
    run = run_benchmark(
        "teste",
        configs,
        items,
        provider_factories=_mock_factories(["olá pessoal bem vindo", "hoje vamos aprender"], ("bad",)),
    )

    ok, bad = run.configs
    assert ok.status == "ok" and bad.status == "error"
    with_ref, without_ref = ok.items
    assert with_ref.wer is not None and with_ref.wer.substitutions == 1
    assert without_ref.wer is None
    assert with_ref.real_time_factor is not None and with_ref.audio_duration_sec > 0
    assert all(r.status == "error" for r in bad.items)
    assert run.averages[0].wer.reference_words == 7  # só o item com referência

    markdown = format_markdown(run)
    assert "| `mock/ok:cpu:int8`" in markdown
    assert "## Média geral" in markdown
    assert "## Falhas" in markdown
    assert "aprender" in markdown  # substituição listada
    sem_ref_row = [l for l in markdown.split("## sem-ref")[1].splitlines() if "mock/ok" in l][0]
    assert "—" in sem_ref_row

    # results.json validado
    reloaded = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert reloaded.configs[0].items[0].wer.substitutions == 1


def test_single_item_has_no_average_section(tmp_path):
    items = [BenchmarkItem(name="um", audio=_write_audio(tmp_path / "a.wav"))]
    run = run_benchmark(
        "x", [BenchmarkConfig.parse("mock/m:cpu:int8")], items, provider_factories=_mock_factories(["a b"])
    )
    assert "## Média geral" not in format_markdown(run)


def test_unknown_provider_is_recorded(tmp_path):
    items = [BenchmarkItem(name="um", audio=_write_audio(tmp_path / "a.wav"))]
    run = run_benchmark("x", [BenchmarkConfig.parse("nada/m:cpu:int8")], items)
    assert run.configs[0].status == "error" and "desconhecido" in run.configs[0].error


def test_manifest_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "sub" / "m.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps({"items": [{"name": "a", "audio": "a.wav", "reference": "a.txt", "language": "en"}, {"name": "b", "audio": "b.wav"}]}),
        encoding="utf-8",
    )
    loaded = BenchmarkManifest.load(manifest)
    assert loaded.items[0].audio == (tmp_path / "sub").resolve() / "a.wav"
    assert loaded.items[0].reference.name == "a.txt" and loaded.items[0].language == "en"
    assert loaded.items[1].reference is None


def test_manifest_duplicate_names_rejected(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"name": "a", "audio": "x"}, {"name": "a", "audio": "y"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        BenchmarkManifest.load(manifest)


def test_cli_mock_end_to_end(tmp_path, capsys, monkeypatch):
    audio = _write_audio(tmp_path / "aula.wav")
    monkeypatch.setattr("sys.argv", [
        "x", "benchmark", "transcription", "--audio", str(audio),
        "--configs", "mock/m:cpu:int8", "--results-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    out = list((tmp_path / "out").glob("*-aula"))
    assert len(out) == 1
    assert (out[0] / "results.json").is_file() and (out[0] / "results.md").is_file()


def test_cli_requires_audio_or_manifest():
    args = build_parser().parse_args(["benchmark", "transcription", "--configs", "small:cpu:int8"])
    assert args.audio is None and args.manifest is None
