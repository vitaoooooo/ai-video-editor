"""Mapa de assuntos (1ª chamada do Diretor de cobertura): blocos de frases seguidas que falam da mesma
coisa, no mesmo lugar/momento.

A IA divide; o código valida o mapa inteiro (blocos em ordem, contíguos, sem buraco nem sobreposição,
cobrindo todas as frases mantidas, com assunto). Inválido → nova tentativa com os problemas; de novo
inválido (ou IA desligada) → o mapa do código: um bloco por trecho seguido do mesmo clipe. Um mapa pela
metade não serve (os respiros dependem das trocas de bloco), por isso a recusa é do mapa, não do bloco.
"""

import time
from typing import Optional

from ai_video_editor.coverage.provider import CoverageDirector
from ai_video_editor.coverage.schema import CoverageContext, CoverageRun, CoverageSentence, Topic, TopicMap
from ai_video_editor.director.service import record_effective_model
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.schema import AIRunMetrics

MAX_TOPIC_ATTEMPTS = 2


def _topic(n: int, sentences: list[CoverageSentence], i: int, j: int, topic: str, place: str) -> Topic:
    return Topic(
        id=f"A{n}", from_sentence=sentences[i].id, to_sentence=sentences[j].id, topic=topic.strip(), place=place.strip(),
        start_frame=sentences[i].start_frame, end_frame=sentences[j].end_frame,
    )


def topics_by_clip(sentences: list[CoverageSentence]) -> list[Topic]:
    """Mapa do código: um bloco por trecho seguido do mesmo clipe (o clipe costuma ser um lugar/momento)."""
    topics: list[Topic] = []
    start = 0
    for k in range(1, len(sentences) + 1):
        if k == len(sentences) or sentences[k].clip_id != sentences[start].clip_id:
            topics.append(_topic(len(topics) + 1, sentences, start, k - 1, f"clipe {sentences[start].clip_id}", ""))
            start = k
    return topics


def validate_topics(answer: TopicMap, sentences: list[CoverageSentence]) -> tuple[Optional[list[Topic]], list[str]]:
    """(blocos validados, []) ou (None, problemas)."""
    order = {s.id: i for i, s in enumerate(sentences)}
    problems: list[str] = []
    if not answer.blocks:
        return None, ["nenhum bloco"]
    expected = 0
    topics: list[Topic] = []
    for n, block in enumerate(answer.blocks, start=1):
        label = f"bloco {n} ({block.from_sentence}–{block.to_sentence})"
        missing = [x for x in (block.from_sentence, block.to_sentence) if x not in order]
        if missing:
            problems.append(f"{label}: a frase {missing[0]} não existe (ou foi removida do vídeo)")
            continue
        i, j = order[block.from_sentence], order[block.to_sentence]
        if i > j:
            problems.append(f"{label}: {block.from_sentence} vem depois de {block.to_sentence}")
            continue
        if not block.topic.strip():
            problems.append(f"{label}: sem assunto")
        if i > expected:
            gap = sentences[expected].id if i - 1 == expected else f"{sentences[expected].id}–{sentences[i - 1].id}"
            problems.append(f"buraco: {gap} sem bloco (antes do {label})")
        elif i < expected:
            overlap = sentences[i].id if min(j, expected - 1) == i else f"{sentences[i].id}–{sentences[min(j, expected - 1)].id}"
            problems.append(f"sobreposição: {overlap} já estão no bloco anterior ({label})")
        topics.append(_topic(len(topics) + 1, sentences, i, j, block.topic, block.place))
        expected = max(expected, j + 1)
    if expected < len(sentences):
        rest = sentences[expected].id if expected == len(sentences) - 1 else f"{sentences[expected].id}–{sentences[-1].id}"
        problems.append(f"buraco: {rest} sem bloco (no fim)")
    return (None, problems) if problems else (topics, [])


def run_topics(director: Optional[CoverageDirector], context: CoverageContext, run: CoverageRun) -> list[Topic]:
    """Pede o mapa de assuntos (com uma nova tentativa só com os problemas) e registra em `run`."""
    sentences = context.sentences
    if director is None or not sentences:
        run.topics_origin = "fallback"
        run.topics_reason = "IA de cobertura desligada: um bloco por clipe" if director is None else "nenhuma frase mantida"
        return topics_by_clip(sentences)
    ask = context
    for attempt in range(1, MAX_TOPIC_ATTEMPTS + 1):
        director.last_input_tokens = None
        director.last_output_tokens = None
        director.last_api_attempts = 1
        started = time.perf_counter()
        answer: Optional[TopicMap] = None
        problems: list[str] = []
        error: Optional[AIResponseValidationError] = None
        try:
            answer = director.map_topics(ask)
        except AIResponseValidationError as exc:
            error = exc
            problems = [exc.details or str(exc)]
        run.topic_responses.append(answer)
        topics = None
        if answer is not None:
            topics, problems = validate_topics(answer, sentences)
        record_effective_model(run, director)
        run.topic_attempts.append(
            AIRunMetrics(
                provider=run.effective_provider or director.provider_name,
                model_id=run.effective_model_id or director.model_id,
                frames_sent=0,
                latency_sec=round(time.perf_counter() - started, 3),
                input_tokens=director.last_input_tokens,
                output_tokens=director.last_output_tokens,
                api_attempts=director.last_api_attempts,
                success=topics is not None,
                error_type=type(error).__name__ if error else (None if topics is not None else "MapaInvalido"),
            )
        )
        if topics is not None:
            run.topics_origin = "ia"
            run.video_summary = answer.summary.strip()
            return topics
        ask = context.model_copy(update={"previous_error": "\n".join(f"- {p}" for p in problems)})
    run.topics_origin = "fallback"
    run.topics_reason = f"mapa de assuntos inválido em {MAX_TOPIC_ATTEMPTS} tentativas ({problems[0]}...): um bloco por clipe"
    run.warnings.append(f"mapa de assuntos da IA recusado ({'; '.join(problems[:3])}); usado um bloco por clipe")
    return topics_by_clip(sentences)
