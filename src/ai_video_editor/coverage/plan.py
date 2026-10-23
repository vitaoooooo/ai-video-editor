"""Plano da cobertura em linguagem de gente: uma linha por bloco de assunto (terminal, relatório e
`coverage_plan.md`).

Ex.: `S034–S041 · trilha até o lago · coberto 2× (D001.T10 'lago' ↔ 'lago escuro entre montanhas')` ou
`S000–S012 · chegada de ônibus · sem cobertura (nenhuma imagem de ônibus)`.
"""

import re
from collections import Counter

from ai_video_editor.coverage.schema import CoverageContext, CoverageRun

# Coberturas detalhadas por linha (as outras viram "+N").
SHOWN_PER_BLOCK = 2
PLAN_FILE = "coverage_plan.md"

_SENTENCE = re.compile(r"\bS\d{3,}\b")


def _quote(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


def plan_lines(run: CoverageRun, context: CoverageContext) -> list[str]:
    order = {s.id: i for i, s in enumerate(context.sentences)}
    fps = context.fps
    spans = {s.id: s for s in run.bad_spans}
    outcomes = {o.span: o for o in run.span_outcomes}
    lines = []
    for topic in run.topics:
        lo, hi = order[topic.from_sentence], order[topic.to_sentence]
        inside = [c for c in run.coverages if topic.start_frame <= c.start_frame < topic.end_frame]
        content = [c for c in inside if c.kind == "conteudo"]
        patches = [c for c in inside if c.kind == "tela_uniforme"]
        parts = []
        if content:
            shown = [
                f"{c.shot_ids[0]} '{_quote(c.evidence.speech)}' ↔ '{_quote(c.evidence.image)}'"
                for c in content[:SHOWN_PER_BLOCK]
            ]
            more = f"; +{len(content) - SHOWN_PER_BLOCK}" if len(content) > SHOWN_PER_BLOCK else ""
            parts.append(f"coberto {len(content)}× ({'; '.join(shown)}{more})")
        if patches:
            shots = ", ".join(x for c in patches for x in c.shot_ids)
            parts.append(f"tela uniforme tapada ({shots}, relacionada)")
        if not parts:
            dropped = [
                d for d in run.discarded
                if d.item.startswith("cobertura") and (m := _SENTENCE.search(d.item)) and m.group(0) in order and lo <= order[m.group(0)] <= hi
            ]
            if dropped:
                kinds = Counter(d.kind for d in dropped)
                why = ", ".join(f"{k}: {n}" if len(kinds) > 1 else k for k, n in kinds.most_common())
                parts.append(f"sem cobertura ({len(dropped)} proposta(s) descartada(s): {why})")
            elif topic.place:
                parts.append(f"sem cobertura (nenhuma imagem de {topic.place})")
            else:
                parts.append("sem cobertura (nenhuma imagem que combine com a fala)")
        for b in run.breaths:
            if b.after_sentence == topic.to_sentence:
                parts.append(f"respiro depois ({b.shot_id})")
        for span in spans.values():
            o = outcomes.get(span.id)
            if o and o.status != "coberto" and topic.start_frame <= span.start_frame < topic.end_frame:
                what = "imagem ruim mantida" if o.status == "mantido" else "imagem ruim em parte"
                parts.append(f"{what} ({span.id}, {_fmt(span.duration_sec(fps))})")
        lines.append(f"{topic.span_label} · {topic.topic} · {' · '.join(parts)}")
    return lines


def plan_markdown(run: CoverageRun, project_name: str) -> str:
    """`coverage_plan.md`: o plano por bloco e o resumo (o mesmo do terminal)."""
    lines = [f"# Plano da cobertura — {project_name}", ""]
    if run.video_summary:
        lines += [f"**Do que o vídeo trata:** {run.video_summary}", ""]
    lines += [
        f"Mapa de assuntos: {'IA' if run.topics_origin == 'ia' else 'código (um bloco por clipe)'}"
        + (f" — {run.topics_reason}" if run.topics_reason else ""),
        f"Cobertura de conteúdo: {_fmt(run.covered_sec)} de {_fmt(run.spine_duration_sec)}; "
        f"imagem ruim: {run.bad_label}; origem {run.origin_label}",
        "",
        *(f"- {line}" for line in run.plan),
        "",
    ]
    if run.warnings:
        lines += ["## Avisos", "", *(f"- {w}" for w in run.warnings), ""]
    return "\n".join(lines)
