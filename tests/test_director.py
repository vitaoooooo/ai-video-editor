"""Testes do Diretor de IA: montagem de frases, validação, nova tentativa e provedores (sem rede)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_video_editor.director import (
    DirectorContext,
    DirectorDecision,
    SentenceRemoval,
    UserBrief,
    build_sentences,
    run_director,
    validate_decision,
)
from ai_video_editor.director.providers.gemini import GeminiDirector, build_director_prompt
from ai_video_editor.director.providers.mock import MockDirector
from ai_video_editor.director.schema import Sentence
from ai_video_editor.director.sentences import split_by_removal
from ai_video_editor.intelligence.exceptions import AIRateLimitError, AIResponseValidationError
from ai_video_editor.transcription.schema import Transcript, TranscriptSegment


def _transcript(segments: list[tuple[float, float, str]]) -> Transcript:
    return Transcript(
        source_path=Path("aula.mp4"),
        language="pt",
        duration_sec=segments[-1][1] + 1.0,
        model_used="mock",
        segments=[
            TranscriptSegment(id=i, start_sec=start, end_sec=end, text=text)
            for i, (start, end, text) in enumerate(segments)
        ],
    )


def _sentences(*durations: float) -> list[Sentence]:
    """Frases de durações dadas, separadas por 0,5s."""
    out, cursor = [], 0.0
    for i, d in enumerate(durations):
        out.append(Sentence(id=i, start_sec=cursor, end_sec=cursor + d, text=f"frase {i}.", segment_ids=[i]))
        cursor += d + 0.5
    return out


def _removal(sentence_id: int, category: str = "repetition") -> SentenceRemoval:
    return SentenceRemoval(sentence_id=sentence_id, category=category, reason="motivo")


CONTEXT = DirectorContext(video_name="aula.mp4", duration_sec=60.0, language="pt")


# --- montagem de frases ---


def test_sentences_join_segments_until_final_punctuation():
    transcript = _transcript(
        [
            (0.0, 2.0, " Hoje vamos aprender"),
            (2.1, 3.5, " a editar vídeo."),
            (3.8, 5.0, " Isso é fácil?"),
            (5.2, 6.0, " Sim"),
            (6.1, 7.0, " muito!"),
        ]
    )
    sentences = build_sentences(transcript)
    assert [s.text for s in sentences] == [
        "Hoje vamos aprender a editar vídeo.",
        "Isso é fácil?",
        "Sim muito!",
    ]
    assert [s.id for s in sentences] == [0, 1, 2]
    assert [s.segment_ids for s in sentences] == [[0, 1], [2], [3, 4]]
    assert (sentences[0].start_sec, sentences[0].end_sec) == (0.0, 3.5)
    assert (sentences[2].start_sec, sentences[2].end_sec) == (5.2, 7.0)


def test_sentences_split_on_long_pause_without_punctuation():
    transcript = _transcript(
        [
            (0.0, 2.0, " então a gente abre"),
            (2.5, 3.0, " o menu"),  # pausa de 0,5s: mesma frase
            (4.2, 6.0, " depois salva o arquivo"),  # pausa de 1,2s: nova frase
            (6.3, 7.0, " no final"),  # sem pontuação no fim: fecha a frase
        ]
    )
    sentences = build_sentences(transcript)
    assert [s.segment_ids for s in sentences] == [[0, 1], [2, 3]]
    assert sentences[1].text == "depois salva o arquivo no final"


def test_sentences_pause_threshold_is_strict_and_configurable():
    transcript = _transcript([(0.0, 1.0, "a"), (2.0, 3.0, "b"), (4.5, 5.0, "c")])
    assert [s.segment_ids for s in build_sentences(transcript)] == [[0, 1], [2]]
    assert [s.segment_ids for s in build_sentences(transcript, max_pause_sec=2.0)] == [[0, 1, 2]]


def test_split_by_removal_merges_consecutive_removed_sentences():
    sentences = _sentences(2, 2, 2, 2, 2)
    cuts, kept = split_by_removal(sentences, {1, 2, 4})
    assert [(c.start_sec, c.end_sec) for c in cuts] == [(2.5, 7.0), (10.0, 12.0)]
    assert [(k.start_sec, k.end_sec) for k in kept] == [(0.0, 2.0), (7.5, 9.5)]


# --- validação ---


def test_validate_accepts_valid_decision():
    validate_decision(DirectorDecision(removals=[_removal(1)], summary="ok"), _sentences(5, 5, 5), UserBrief())


def test_validate_rejects_unknown_id():
    decision = DirectorDecision(removals=[_removal(7)], summary="x")
    with pytest.raises(AIResponseValidationError) as exc:
        validate_decision(decision, _sentences(5, 5), UserBrief())
    assert "7 não existe" in exc.value.details


def test_validate_rejects_duplicated_id():
    decision = DirectorDecision(removals=[_removal(1), _removal(1, "filler")], summary="x")
    with pytest.raises(AIResponseValidationError) as exc:
        validate_decision(decision, _sentences(5, 5, 5), UserBrief())
    assert "mais de uma vez" in exc.value.details


def test_validate_rejects_filler_when_user_keeps_fillers():
    decision = DirectorDecision(removals=[_removal(0, "filler")], summary="x")
    validate_decision(decision, _sentences(1, 5, 5), UserBrief())
    with pytest.raises(AIResponseValidationError) as exc:
        validate_decision(decision, _sentences(1, 5, 5), UserBrief(remove_fillers=False))
    assert "não remover enrolação" in exc.value.details


def test_validate_rejects_excessive_removal_unless_target_duration_requires_it():
    sentences = _sentences(10, 10, 10, 10, 10)  # 50s de fala
    decision = DirectorDecision(removals=[_removal(0), _removal(1), _removal(2)], summary="x")  # 60%
    with pytest.raises(AIResponseValidationError) as exc:
        validate_decision(decision, sentences, UserBrief())
    assert "passa do limite de 40%" in exc.value.details

    # 40% exatos é permitido
    validate_decision(DirectorDecision(removals=[_removal(0), _removal(1)], summary="x"), sentences, UserBrief())
    # Duração alvo de 25s pede ~50%: o limite sobe para 60%
    validate_decision(decision, sentences, UserBrief(target_duration_sec=25))
    # Duração alvo folgada não aumenta o limite
    with pytest.raises(AIResponseValidationError):
        validate_decision(decision, sentences, UserBrief(target_duration_sec=45))


# --- execução com nova tentativa e fallback ---


def test_run_director_success_records_metrics():
    director = MockDirector(removals=[_removal(1)])
    run = run_director(director, _sentences(5, 5, 5), UserBrief(), CONTEXT)
    assert run.decision is not None and run.warnings == []
    assert [s.id for s, _ in run.removed_sentences] == [1]
    assert len(run.attempts) == 1 and run.attempts[0].success
    assert run.attempts[0].provider == "mock" and run.attempts[0].frames_sent == 0


def test_run_director_retries_once_sending_the_error():
    bad = DirectorDecision(removals=[_removal(9)], summary="ruim")
    good = DirectorDecision(removals=[_removal(0)], summary="boa")
    director = MockDirector(responses=[bad, good])
    run = run_director(director, _sentences(5, 5, 5), UserBrief(), CONTEXT)

    assert run.decision == good
    assert len(director.calls) == 2
    assert director.calls[0].previous_error is None
    assert "9 não existe" in director.calls[1].previous_error
    assert [a.success for a in run.attempts] == [False, True]
    assert run.attempts[0].error_type == "AIResponseValidationError"


def test_run_director_falls_back_after_second_invalid_response():
    bad = DirectorDecision(removals=[_removal(0), _removal(1), _removal(2)], summary="tudo")
    parse_error = AIResponseValidationError("gemini", details="JSON inválido")
    director = MockDirector(responses=[parse_error, bad])
    run = run_director(director, _sentences(5, 5, 5), UserBrief(), CONTEXT)

    assert run.decision is None
    assert run.removed_sentences == []
    assert len(director.calls) == 2
    assert len(run.warnings) == 1
    assert "só o corte de silêncio" in run.warnings[0]
    assert "JSON inválido" in run.warnings[0] and "passa do limite" in run.warnings[0]


def test_run_director_does_not_swallow_provider_errors():
    director = MockDirector(responses=[AIRateLimitError("gemini")])
    with pytest.raises(AIRateLimitError):
        run_director(director, _sentences(5), UserBrief(), CONTEXT)


def test_mock_director_default_removes_repeated_sentences():
    transcript = _transcript(
        [(0, 2, "Vamos abrir o editor."), (3, 5, "vamos abrir o editor!"), (6, 8, "Pronto.")]
    )
    decision = MockDirector().decide(build_sentences(transcript), UserBrief(), CONTEXT)
    assert [(r.sentence_id, r.category) for r in decision.removals] == [(1, "repetition")]
    assert decision.removals[0].reason == "Repete a frase 0."


# --- Gemini com cliente falso ---


class _FakeModels:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_token_count=321, candidates_token_count=45)
        return SimpleNamespace(text=self.texts.pop(0), usage_metadata=usage)


def _fake_client(*texts: str):
    return SimpleNamespace(models=_FakeModels(list(texts)))


def test_gemini_director_prompt_and_json_parse():
    sentences = build_sentences(
        _transcript(
            [
                (0.0, 2.0, " Então a gente vai, a gente vai"),
                (2.2, 4.0, " Então vamos abrir o projeto."),
                (5.0, 7.0, " Clique em salvar."),
            ]
        )
    )
    payload = {
        "removals": [{"sentence_id": 0, "category": "false_start", "reason": "Frase recomeçada logo depois."}],
        "summary": "Removida uma frase recomeçada.",
    }
    client = _fake_client(json.dumps(payload))
    director = GeminiDirector(model_id="gemini-2.5-pro", client=client)
    brief = UserBrief(intensity="medium", target_duration_sec=50, instructions="Mantenha as piadas")

    decision = director.decide(sentences, brief, CONTEXT)

    assert decision.removals[0].sentence_id == 0
    assert decision.removals[0].category == "false_start"
    assert decision.summary == "Removida uma frase recomeçada."
    assert (director.last_input_tokens, director.last_output_tokens) == (321, 45)

    call = client.models.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    assert call["config"].response_schema is DirectorDecision
    assert call["config"].response_mime_type == "application/json"
    prompt = call["contents"]
    assert "Na dúvida, MANTENHA" in prompt
    assert "informação nova" in prompt
    assert "em português" in prompt
    assert "SOMENTE pelo ID" in prompt
    assert "MÉDIA" in prompt
    assert "cerca de 50s" in prompt
    assert "Mantenha as piadas" in prompt
    assert "[0] (4.0s) Então a gente vai, a gente vai Então vamos abrir o projeto." in prompt
    assert "[1] (2.0s) Clique em salvar." in prompt
    assert "rejeitada" not in prompt


def test_gemini_director_prompt_includes_previous_error_and_filler_ban():
    sentences = _sentences(3, 3)
    context = CONTEXT.model_copy(update={"previous_error": "sentence_id 9 não existe"})
    prompt = build_director_prompt(sentences, UserBrief(remove_fillers=False), context)
    assert "rejeitada" in prompt and "sentence_id 9 não existe" in prompt
    assert "'filler' está proibida" in prompt
    assert "LEVE" in prompt


def test_gemini_director_invalid_json_triggers_retry_and_fallback():
    client = _fake_client("não é json", json.dumps({"removals": [{"sentence_id": 0}], "summary": "x"}))
    director = GeminiDirector(client=client)
    run = run_director(director, _sentences(3, 3), UserBrief(), CONTEXT)
    assert run.decision is None and run.warnings
    assert len(client.models.calls) == 2
    assert "rejeitada" in client.models.calls[1]["contents"]
    assert run.attempts[0].input_tokens == 321


def test_gemini_director_maps_sdk_errors():
    class Broken:
        def generate_content(self, **kwargs):
            raise RuntimeError("HTTP 429 quota exceeded")

    director = GeminiDirector(client=SimpleNamespace(models=Broken()))
    with pytest.raises(AIRateLimitError):
        director.decide(_sentences(3), UserBrief(), CONTEXT)
