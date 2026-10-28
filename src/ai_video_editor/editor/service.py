"""Passadas do Editor de IA: resumos por clipe (E9), estrutura (E10) e, por bloco, as três perguntas da
passada 2 (E11): 2a cortes por clipe, 2b inclusões e respiros, 2c coberturas. O plano do bloco é montado
pelo código a partir das três respostas (ordem do roteiro; sem pergunta de ordem).

Cada chamada aceita vai para um cache em disco assim que termina (um Ctrl+C no meio não perde o que já
foi feito), em `<projects>/_cache/editor_ia/<passada>/<fingerprint>.json`, com fingerprint = o pedido
exato (sem a lista de erros de uma nova tentativa) + versão do pedido da passada + provedor, cadeia de
modelos e raciocínio. Como o pedido do resumo só contém a seção do clipe no roteiro, mudar o pedido de
outra passada não refaz os resumos. Respostas inaproveitáveis não vão para o cache.

Alvo do vídeo (`_resolve_target`): a duração do briefing/CLI; sem ela, a escrita no texto do pedido
(`material.request_duration`: "uns 8 minutos" = 480 s); sem nenhuma, a que a estrutura escolheu. O do pedido
acima da fala do material inteiro vira a fala disponível (com aviso). As metas que a IA deu aos blocos são
reescaladas para somar o alvo, na proporção dela (`_rescale_blocks`).

Meta de cada clipe na 2a (aritmética, sem decisão; `_clip_targets`): clipes sem pergunta de cortes (uma frase
ou fala até `CUT_MIN_SPEECH_SEC`) entram inteiros e contam a fala toda; o resto da meta do bloco é repartido
pela fala livre de cada clipe, nunca acima da fala do clipe (o excesso vai para os outros); um clipe dividido
em partes reparte a sua meta do mesmo jeito.

A estrutura não precisa usar todos os clipes: um clipe com fala que a IA não pôs em bloco nem em "fora"
conta como fora ("não escolhido pela IA"), sem bloco extra. Falta de duração (medida, sem decisão): se a fala
dos clipes nos blocos fica abaixo de `STRUCTURE_MIN_RATIO` da duração do pedido (a que a IA escolhe sozinha
não conta: os dois números seriam dela) e há clipes fora dos blocos, UMA nova tentativa informa os números e lista os clipes com fala não
usados e o acervo; a IA pode incluir clipes ou manter a resposta (a resposta igual é aceita).

2b (medida, sem decisão): sobra do bloco = alvo do bloco − fala mantida depois da 2a. Bloco com fala sem
sobra (ou sem alvo) não tem pergunta 2b. Com sobra, o pedido informa a sobra, as posições medidas
(`validate.insert_positions`: respiro só no começo de um bloco que não é o 1º, ou numa troca de lugar, nunca
onde a fala continua) e até `INSERT_CANDIDATES` imagens: primeiro as sem fala do próprio bloco, depois as do
acervo com afinidade medida com os trechos vizinhos das posições de respiro (`material.breath_affinity`),
depois o resto. 2c: só trechos ditos para a câmera (`validate.talking_sentences`) recebem candidatos.

Nova tentativa (uma, com a lista de erros): estrutura sem bloco válido; qualquer pergunta da passada 2 com
metade ou mais dos itens inválida; 2a com todas as frases cortadas sem `fora: true` (`validate.ALL_CUT`) ou
com a duração fora da meta ± `DURATION_TOLERANCE` (a nova tentativa mostra os números); 2b de bloco sem fala
que não escolhe nada. Depois dela, a última resposta aproveitável é aceita com os itens válidos; na 2a, entre
as respostas aproveitáveis, a mais perto da meta (`_closest_cut`; uma que corta tudo nunca ganha de uma que
mantém algo; duração fora da meta é registrada). Só a estrutura ou um `fora: true` explícito tiram um clipe:
se todas as respostas cortam tudo sem `fora: true`, o clipe entra inteiro com aviso (`ALL_CUT_KEPT`). A nova
tentativa da 2a por duração diz quantos segundos cortar ou devolver e lista as frases com os segundos (acima
da meta, as que ficaram; abaixo, as cortadas). Na 2a, os
pares contínuos (`validate.continuous_groups`) vão marcados no roteiro, e um par cortado pela metade é
ajustado pela validação (corta ou fica junto). Fallback neutro
(com aviso) só sem resposta aproveitável: 2a = nada cortado;
2b = nada a mais (bloco sem fala: todas as imagens dele, em ordem); 2c = sem cobertura. Depois de
`MAX_CONSECUTIVE_FAILURES` erros seguidos da API (não de validação), a IA não é mais chamada nesta execução
e o resto vai pelos fallbacks. Erro de autenticação interrompe tudo.

Cada pedido fica registrado em `EditorRun.requests` com os tokens estimados pelo orçamento e os reais
informados pelo provedor (acompanhamento da calibração de `budget.TOKEN_FACTOR`).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from ai_video_editor.editor.budget import Budget, tokens
from ai_video_editor.editor.ids import IdResolver
from ai_video_editor.editor.material import (
    EditorMaterial,
    asks_landscape_opening,
    breath_affinity,
    coverage_matches,
    library_ranking,
    request_duration,
)
from ai_video_editor.editor.prompt import (
    COVERAGE_PROMPT_VERSION,
    CUTS_PROMPT_VERSION,
    INSERTS_PROMPT_VERSION,
    SHORTFALL_SUMMARY_WORDS,
    STRUCTURE_PROMPT_VERSION,
    STRUCTURE_SUMMARY_WORDS,
    SUMMARY_PROMPT_VERSION,
    TRECHO_WORDS,
    BlockHeader,
    CoveragePromptInput,
    CoverageQuestion,
    CutsPromptInput,
    EditorContext,
    InsertsPromptInput,
    StructureShortfall,
    build_coverage_prompt,
    build_cuts_prompt,
    build_inserts_prompt,
    build_structure_prompt,
    build_summary_prompt,
    duration_retry_line,
    sentence_retry_lines,
    short_text,
)
from ai_video_editor.editor.provider import (
    CoverageRequest,
    CutsRequest,
    EditorAI,
    InsertsRequest,
    StructureRequest,
    SummaryRequest,
)
from ai_video_editor.editor.schema import (
    COVERAGE_PLACE,
    COVERAGE_SAYS,
    Block,
    CallStats,
    Candidate,
    ClipSummary,
    CutAttempt,
    ClipSummaryAnswer,
    CoveragesAnswer,
    CutRun,
    CutsAnswer,
    DetailRun,
    EditorRun,
    InsertsAnswer,
    KeepRange,
    RequestRecord,
    Skipped,
    StructureAnswer,
    StructureRun,
    SubpassRun,
    SummaryNote,
    Trecho,
)
from ai_video_editor.editor.validate import (
    NOT_CHOSEN,
    RESPIRO_MAX_SEC,
    Usage,
    build_trechos,
    continuing_sentences,
    insert_positions,
    letters,
    neutral_inserts,
    neutral_structure,
    sentences_sec,
    speech_places,
    take_sentences,
    talking_sentences,
    trecho_place,
    validate_coverages,
    validate_cuts,
    validate_inserts,
    validate_structure,
)
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIResponseValidationError,
)
from ai_video_editor.pipeline.project import fingerprint
from ai_video_editor.pipeline.schema import StageProgress
from ai_video_editor.roteiro.report import clock
from ai_video_editor.roteiro.schema import Moment

EDITOR_CACHE_DIR = Path("_cache") / "editor_ia"
# Tentativas por pedido: a 1ª + uma nova com a lista de erros.
MAX_ATTEMPTS = 2
# Erros seguidos da API (não de validação) antes de parar de chamar a IA nesta execução.
MAX_CONSECUTIVE_FAILURES = 3
# Faixa aceita da duração resultante de um clipe na 2a: meta ± 35%. Fora dela, UMA nova tentativa com os
# números; se ainda ficar fora, aceita e registra (a duração é informação para a IA, não decisão do código).
DURATION_TOLERANCE = 0.35
# Clipe com fala livre até isto (ou com uma frase só) não tem pergunta de cortes: entra inteiro. Medido no
# vlog real: clipes de uma frase (3,6 a 31,6 s) eram cortados inteiros com o motivo copiado do formato, e um
# clipe curto tem pouco o que cortar sem perder o sentido (a estrutura já decidiu que ele entra).
CUT_MIN_SPEECH_SEC = 12.0
# Aviso de quando todas as respostas da 2a cortaram tudo sem fora: true.
ALL_CUT_KEPT = "IA cortou tudo sem dizer fora; mantido inteiro"
# Estrutura com a fala dos clipes escolhidos abaixo desta fração da meta: UMA nova tentativa informa quanto
# falta e o que não foi usado (a IA pode manter a resposta).
STRUCTURE_MIN_RATIO = 0.85
# Candidatos de cobertura listados por trecho na 2c.
COVERAGE_CANDIDATES = 4
# Candidatos com letra na 2b de um bloco com fala (imagens do bloco primeiro, depois o acervo). Num bloco sem
# fala, as imagens do próprio bloco vão todas (elas são o bloco) e o acervo só preenche até este número.
INSERT_CANDIDATES = 6
# Descrição de um candidato da 2b: até tantas palavras.
CANDIDATE_WORDS = 14
# Motivo de um trecho sem pergunta de cobertura.
NOT_TALKING = "imagem não é fala para câmera"
# Texto de um trecho na 2c: até tantas palavras (a palavra em comum é conferida na fala inteira).
COVERAGE_TRECHO_WORDS = 60

Notice = Callable[[str], None]
Progress = Callable[[StageProgress], None]


class CallCache:
    """Respostas aceitas por passada e fingerprint (gravadas na hora)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    def load(self, kind: str, key: str, schema: type[BaseModel]):
        path = self.path(kind, key)
        if not path.is_file():
            return None
        try:
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def save(self, kind: str, key: str, answer: BaseModel) -> None:
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(answer.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


def editor_cache(projects_root: Path) -> CallCache:
    return CallCache(Path(projects_root) / EDITOR_CACHE_DIR)


@dataclass
class Decision:
    """Resultado de uma pergunta da passada 2: a conferência aceita (None = fallback) e o caminho até ela."""

    check: object = None
    answer: Optional[BaseModel] = None
    origin: str = "ia"
    problem: Optional[str] = None  # problema extra (duração) que persistiu depois da nova tentativa
    attempts: list[str] = field(default_factory=list)
    estimated: int = 0
    tries: list[tuple[int, object]] = field(default_factory=list)  # (tentativa, conferência) aproveitáveis
    chosen: Optional[int] = None  # a tentativa usada


def _fmt_sec(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + "s"


class EditorService:
    """Executa as passadas sobre o material e devolve o `EditorRun` (sem tempos de mídia: isso é a montagem)."""

    def __init__(
        self,
        ai: EditorAI,
        material: EditorMaterial,
        context: EditorContext,
        budget: Budget,
        cache: CallCache,
        on_notice: Optional[Notice] = None,
        on_progress: Optional[Progress] = None,
    ):
        self.ai = ai
        self.material = material
        self.context = context
        self.budget = budget
        self.cache = cache
        self.on_notice = on_notice
        self.on_progress = on_progress
        self.failures = 0
        self.gave_up = False
        self.calls_done = 0
        self.calls_total = 0
        self.requests: list[RequestRecord] = []

    # --- infraestrutura ---

    @property
    def identity(self) -> list:
        return [self.ai.provider_name, list(self.ai.model_chain), self.ai.thinking_label]

    def key(self, kind: str, version: str, prompt: str) -> str:
        return fingerprint("editor", kind, version, prompt, self.identity)

    def _notice(self, message: str) -> None:
        if self.on_notice is not None:
            self.on_notice(message)

    def _progress(self, note: str) -> None:
        if self.on_progress is not None:
            self.on_progress(StageProgress(done=self.calls_done, total=max(self.calls_total, 1), unit="calls", note=note))

    def _cached(self, kind: str, unit: str, estimated: int) -> None:
        self.requests.append(RequestRecord(kind=kind, unit=unit, cached=True, estimated_tokens=estimated))

    def _ask(self, call: Callable, request, stats: CallStats, kind: str, unit: str, attempt: int = 1):
        """(resposta, erro). Erro de validação conta como tentativa; erro da API, como falha seguida."""
        if self.gave_up:
            return None, f"a IA não foi chamada ({MAX_CONSECUTIVE_FAILURES} erros seguidos da API antes)"
        record = RequestRecord(kind=kind, unit=unit, attempt=attempt, estimated_tokens=tokens(request.prompt))
        self.ai.last_input_tokens = self.ai.last_output_tokens = None
        started = time.perf_counter()
        error: Optional[str] = None
        answer = None
        try:
            answer = call(request)
            self.failures = 0
        except AIAuthenticationError:
            raise
        except AIResponseValidationError as e:
            error = f"resposta fora do formato: {e}"
            self.failures = 0
        except AIProviderError as e:
            error = f"erro da IA: {e}"
            self.failures += 1
            if self.failures >= MAX_CONSECUTIVE_FAILURES and not self.gave_up:
                self.gave_up = True
                self._notice(
                    f"{MAX_CONSECUTIVE_FAILURES} erros seguidos da IA: o resto vai pelos fallbacks neutros (último: {e})"
                )
        finally:
            stats.calls += 1
            stats.api_calls += max(1, int(getattr(self.ai, "last_api_attempts", 1) or 1))
            for name, attr in (("input_tokens", "last_input_tokens"), ("output_tokens", "last_output_tokens")):
                value = getattr(self.ai, attr, None)
                if isinstance(value, int):
                    setattr(stats, name, (getattr(stats, name) or 0) + value)
                    setattr(record, name, value)
            stats.seconds = round(stats.seconds + time.perf_counter() - started, 3)
            self.requests.append(record)
        return answer, error

    def _decide(
        self,
        kind: str,
        version: str,
        unit: str,
        schema: type[BaseModel],
        build: Callable[[list[str]], str],
        make_request: Callable[[str], object],
        call: Callable,
        check_of: Callable[[BaseModel], object],
        stats: CallStats,
        extra_problem: Optional[Callable[[object], Optional[str]]] = None,
        retry_details: Optional[Callable[[object], list[str]]] = None,
        choose: Optional[Callable[[list], int]] = None,
    ) -> Decision:
        """Uma pergunta: cache → IA → (recusada) nova tentativa com os erros. `check.unusable` decide se a
        resposta serve; `check.refused` e `extra_problem` pedem a nova tentativa; `retry_details` (com o
        `extra_problem`) acrescenta linhas de informação à lista da nova tentativa. Sem `choose`, vale a última
        resposta aproveitável; com ele, a que ele escolher entre as conferências aproveitáveis (em ordem)."""
        prompt = build([])
        decision = Decision(estimated=tokens(prompt))
        key = self.key(kind, version, prompt)
        cached = self.cache.load(kind, key, schema)
        if cached is not None:
            trial = check_of(cached)
            if trial.unusable is None:
                decision.check, decision.answer, decision.origin = trial, cached, "cache"
                decision.problem = extra_problem(trial) if extra_problem else None
                decision.tries, decision.chosen = [(1, trial)], 1
                stats.cached += 1
                self._cached(kind, unit, decision.estimated)
                return decision
        errors: list[str] = []
        details: list[str] = []
        usable: list[tuple[int, object, BaseModel, Optional[str]]] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw, error = self._ask(call, make_request(build(errors)), stats, kind, unit, attempt)
            if raw is not None:
                trial = check_of(raw)
                problem = extra_problem(trial) if extra_problem and trial.unusable is None else None
                if trial.unusable is None:  # aproveitável: vale se a nova tentativa não resolver
                    usable.append((attempt, trial, raw, problem))
                problems = [p for p in (trial.refused, problem) if p]
                if not problems:
                    break
                error = "; ".join(problems) + (": " + "; ".join(trial.errors[:12]) if trial.errors else "")
                if problem and retry_details is not None:
                    details = retry_details(trial)
            errors.append(error)
            errors += details
            details = []
            decision.attempts.append(error)
            if self.gave_up:
                break
        if usable:
            k = choose([u[1] for u in usable]) if choose is not None else len(usable) - 1
            attempt, decision.check, decision.answer, decision.problem = usable[k]
            decision.tries, decision.chosen = [(u[0], u[1]) for u in usable], attempt
            self.cache.save(kind, key, decision.answer)
        return decision

    # --- execução ---

    def run(self) -> EditorRun:
        material = self.material
        run = EditorRun(
            request=self.context.request, target_sec=self.context.target_sec, provider=self.ai.provider_name,
            model_chain=list(self.ai.model_chain), budget=self.budget.info(),
        )
        plans = {clip_id: self._summary_parts(clip_id) for clip_id in material.clip_ids}
        self.calls_total = sum(len(p) for p in plans.values()) + 1
        for clip_id in material.clip_ids:
            run.summaries.append(self._summarize(clip_id, plans[clip_id], run.summary_stats))
        failed = [s.clip_id for s in run.summaries if s.answer is None]
        if failed:
            run.warnings.append(f"Sem resumo (a IA falhou): {', '.join(failed)}; esses clipes foram à estrutura só com os dados")

        run.structure = self._structure({s.clip_id: s for s in run.summaries})
        run.warnings += run.structure.warnings

        usage = Usage()
        self.calls_total = self.calls_done + sum(
            sum(1 for c in b.clip_ids if material.has_speech(c)) + 2 for b in run.structure.blocks
        )
        for block in run.structure.blocks:
            first_block = not any(d.keep for d in run.details)  # o vídeo ainda não começou
            detail = self._block(block, run.structure, usage, first_block)
            run.details.append(detail)
            for cut_run in detail.cut_runs:
                run.cuts_stats.add(cut_run.stats)
            if detail.inserts is not None:
                run.inserts_stats.add(detail.inserts.stats)
            if detail.coverage is not None:
                run.coverage_stats.add(detail.coverage.stats)
            run.warnings += [f"{detail.unit_id}: {w}" for w in detail.warnings]
        run.requests = list(self.requests)
        run.effective_model = (
            f"{self.ai.effective_provider}:{self.ai.effective_model_id}" if self.ai.effective_model_id else None
        )
        run.model_switches = list(self.ai.model_switches)
        return run

    # --- E9: resumos ---

    def _summary_parts(self, clip_id: str) -> list[Optional[list[str]]]:
        """[None] se a seção inteira cabe; senão os momentos de cada pedaço (em ordem)."""
        material = self.material
        available = self.budget.summary_available
        if tokens(build_summary_prompt(material.section(clip_id))) <= available:
            return [None]
        parts: list[list[str]] = []
        current: list[str] = []
        for moment in material.clip_moments.get(clip_id, []):
            trial = current + [moment.id]
            if current and tokens(build_summary_prompt(material.section(clip_id, trial), 2, 2)) > available:
                parts.append(current)
                trial = [moment.id]
            current = trial
        if current:
            parts.append(current)
        return parts or [None]

    def _summarize(self, clip_id: str, parts: list[Optional[list[str]]], stats: CallStats) -> ClipSummary:
        material = self.material
        summary = ClipSummary(clip_id=clip_id, parts=len(parts))
        answers: list[ClipSummaryAnswer] = []
        errors_seen: list[str] = []
        for k, moment_ids in enumerate(parts, start=1):
            unit = clip_id + (f" ({k}/{len(parts)})" if len(parts) > 1 else "")
            self._progress(f"resumo {unit}")
            section = material.section(clip_id, moment_ids)
            prompt = build_summary_prompt(section, k, len(parts))
            key = self.key("resumos", SUMMARY_PROMPT_VERSION, prompt)
            answer = self.cache.load("resumos", key, ClipSummaryAnswer)
            if answer is not None:
                summary.cached_parts += 1
                stats.cached += 1
                self._cached("resumo", unit, tokens(prompt))
            else:
                errors: list[str] = []
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    request = SummaryRequest(
                        clip_id=clip_id, prompt=build_summary_prompt(section, k, len(parts), errors),
                        material=material, moment_ids=moment_ids, part=k, parts=len(parts),
                    )
                    answer, error = self._ask(self.ai.summarize, request, stats, "resumo", unit, attempt)
                    if answer is not None:
                        self.cache.save("resumos", key, answer)
                        break
                    errors.append(error)
                    if self.gave_up:
                        break
                errors_seen += errors if answer is None else []
            self.calls_done += 1
            if answer is not None:
                answers.append(self._clean_summary(clip_id, answer, summary))
            elif len(parts) > 1:
                summary.warnings.append(f"pedaço {k}/{len(parts)} sem resumo")
        if not answers:
            summary.origin, summary.error = "falha", (errors_seen[-1] if errors_seen else "sem resposta")
            return summary
        summary.answer = _join_summaries(answers)
        summary.origin = "cache" if summary.cached_parts == len(parts) else "ia"
        return summary

    def _clean_summary(self, clip_id: str, answer: ClipSummaryAnswer, summary: ClipSummary) -> ClipSummaryAnswer:
        """Normaliza os IDs dos destaques/problemas (`editor/ids.py`) e tira os que não são do clipe (só
        informação para a estrutura); as trocas vão para `summary.normalized_ids`."""
        material = self.material
        known = {s.id for s in material.clip_sentences.get(clip_id, [])} | {m.id for m in material.clip_moments.get(clip_id, [])}
        resolver = IdResolver(material)

        def clean(notes: list[SummaryNote]) -> list[SummaryNote]:
            result = []
            for note in notes:
                ids = [i for i in (resolver.anchor(raw) for raw in note.ids) if i in known]
                if ids or note.motivo:
                    result.append(note.model_copy(update={"ids": ids}))
            return result

        cleaned = answer.model_copy(update={"destaques": clean(answer.destaques), "problemas": clean(answer.problemas)})
        summary.normalized_ids += resolver.fixed
        return cleaned

    # --- E10: estrutura ---

    def structure_lines(self, summaries: dict[str, ClipSummary]) -> list[str]:
        """Uma linha por clipe, todos os clipes (ID, papel, duração, fala, data e o resumo curto)."""
        material = self.material
        return [
            material.structure_line(
                c, summaries[c].answer.resumo if c in summaries and summaries[c].answer else None,
                STRUCTURE_SUMMARY_WORDS,
            )
            for c in material.clip_ids
        ]

    def _structure(self, summaries: dict[str, ClipSummary]) -> StructureRun:
        material, context = self.material, self.context
        lines = self.structure_lines(summaries)
        result = StructureRun(origin="ia")

        def prompt_for(errors: Optional[list[str]] = None, shortfall: Optional[StructureShortfall] = None) -> str:
            return build_structure_prompt(context, lines, errors, shortfall)

        prompt = prompt_for()
        result.estimated_tokens = tokens(prompt)
        self._progress("estrutura")
        answer: Optional[StructureAnswer] = None
        check = None
        if result.estimated_tokens > self.budget.structure_available:
            result.attempts.append(
                f"o pedido da estrutura (~{result.estimated_tokens} tokens, uma linha por clipe) não cabe no "
                f"contexto ({self.budget.num_ctx})"
            )
        else:
            key = self.key("estrutura", STRUCTURE_PROMPT_VERSION, prompt)
            cached = self.cache.load("estrutura", key, StructureAnswer)
            if cached is not None:
                check = validate_structure(cached, material)
                if check.blocks:
                    answer, result.origin = cached, "cache"
                    result.stats.cached += 1
                    self._cached("estrutura", "estrutura", result.estimated_tokens)
            errors: list[str] = []
            shortfall: Optional[StructureShortfall] = None
            first_blocks: list[Block] = []
            for attempt in range(1, 1 + (0 if answer is not None else MAX_ATTEMPTS)):
                request = StructureRequest(
                    prompt=prompt_for(errors, shortfall), material=material, context=context, summaries=summaries,
                )
                raw, error = self._ask(self.ai.structure, request, result.stats, "estrutura", "estrutura", attempt)
                if raw is not None:
                    trial = validate_structure(raw, material)
                    if trial.blocks:  # aproveitável: a última aproveitável vale se a nova tentativa não resolver
                        answer, check = raw, trial
                    if trial.refused is None:
                        if shortfall is None and attempt < MAX_ATTEMPTS:
                            shortfall = self._shortfall(trial, summaries)
                            if shortfall is not None:  # informação, não recusa: a resposta já vale
                                result.speech_sec_first = round(shortfall.speech_sec, 3)
                                result.shortfall = (
                                    f"faltam {round(shortfall.missing_sec)} s: meta {round(shortfall.target_sec)} s, "
                                    f"fala dos clipes escolhidos {round(shortfall.speech_sec)} s"
                                )
                                first_blocks = trial.blocks
                                self._notice(f"estrutura: {result.shortfall}; nova tentativa com os clipes não usados")
                                continue
                        break
                    error = f"{trial.refused}: " + "; ".join(trial.errors[:12])
                errors.append(error)
                result.attempts.append(error)
                if self.gave_up:
                    break
            if first_blocks and check is not None:
                before = {c for b in first_blocks for c in b.clip_ids}
                result.added_clip_ids = [c for b in check.blocks for c in b.clip_ids if c not in before]
            if answer is not None and result.origin != "cache":
                self.cache.save("estrutura", key, answer)
        self.calls_done += 1
        if answer is None:
            answer = neutral_structure(material)
            check = validate_structure(answer, material)
            result.origin = "fallback"
            result.warnings.append(
                "ESTRUTURA PELO FALLBACK NEUTRO (a IA falhou de vez): um bloco por clipe com fala, na ordem do "
                f"catálogo. Motivo: {result.attempts[-1] if result.attempts else 'sem resposta'}"
            )
            self._notice(result.warnings[-1])
        result.blocks, result.out = check.blocks, check.out
        result.discarded = check.discarded
        if result.origin != "fallback":
            result.normalized_ids = check.normalized
        result.notes = answer.notas
        if check.not_chosen:
            self._notice(f"estrutura: fora do vídeo ({NOT_CHOSEN}): {', '.join(check.not_chosen)}")
        if check.discarded and result.origin != "fallback":
            result.warnings.append(f"estrutura: {len(check.discarded)} item(ns) descartado(s) na validação")
        placed = {c for b in result.blocks for c in b.clip_ids} | {o.clipe for o in result.out}
        result.library_clip_ids = [c for c in material.clip_ids if c not in placed]
        in_blocks = {c for b in result.blocks for c in b.clip_ids}
        silent_out = [o.clipe for o in result.out if not material.has_speech(o.clipe)]
        if silent_out and not result.library_clip_ids:
            result.warnings.append(
                f"estrutura: todo o acervo foi para \"fora\" ({', '.join(silent_out)}): a 2b não tem imagens do "
                "acervo para oferecer como respiro"
            )
        result.speech_sec = round(sum(material.speech_sec(c) for c in in_blocks), 3)
        self._resolve_target(result, answer)
        _rescale_blocks(result.blocks, result.target_total_sec, material)
        return result

    def _requested_target(self) -> tuple[Optional[float], str, str]:
        """(alvo, origem, nota) do pedido: a duração do briefing/CLI; sem ela, a escrita no texto do pedido
        (`material.request_duration`)."""
        if self.context.target_sec is not None:
            return self.context.target_sec, "pedido", "duração do briefing ou da CLI"
        found = request_duration(self.context.request)
        if found is not None:
            return found[0], "texto", f"\"{found[1]}\" no texto do pedido"
        return None, "nenhuma", ""

    def _resolve_target(self, result: StructureRun, answer: StructureAnswer) -> None:
        """Alvo do vídeo: o do pedido (`_requested_target`) > o que a estrutura escolheu. O do pedido acima da
        fala disponível no material inteiro vira a fala disponível (com aviso); o da IA fica como ela deu."""
        material = self.material
        target, origin, note = self._requested_target()
        if target is None and answer.duracao_total_alvo_sec and answer.duracao_total_alvo_sec > 0:
            target, origin, note = answer.duracao_total_alvo_sec, "ia", "escolhido pela estrutura (o pedido não diz a duração)"
        available = sum(material.speech_sec(c) for c in material.clip_ids)
        if target is not None and origin != "ia" and 0 < available < target:
            result.target_requested_sec = target
            note += f"; passa da fala disponível no material ({round(available)} s): alvo = a fala disponível"
            result.warnings.append(
                f"alvo: {clock(target)} pedido, mas o material só tem {clock(available)} de fala; alvo = {clock(available)}"
            )
            target = round(available, 1)
        result.target_total_sec, result.target_origin, result.target_note = target, origin, note  # type: ignore[assignment]

    def _shortfall(self, check, summaries: dict[str, ClipSummary]) -> Optional[StructureShortfall]:
        """Falta de duração da estrutura (None = sem duração no pedido, sem fala no material, fala ≥
        `STRUCTURE_MIN_RATIO` da duração pedida ou nada fora dos blocos para oferecer). Só medida: o que
        fazer é da IA."""
        material = self.material
        target = self._requested_target()[0]
        if target is None or target <= 0 or not any(material.has_speech(c) for c in material.clip_ids):
            return None
        in_blocks = {c for b in check.blocks for c in b.clip_ids}
        speech = sum(material.speech_sec(c) for c in in_blocks)
        if speech >= STRUCTURE_MIN_RATIO * target:
            return None
        out_reason = {o.clipe: o.motivo for o in check.out if o.motivo != NOT_CHOSEN}

        def summary(clip_id: str) -> str:
            found = summaries.get(clip_id)
            text = found.answer.resumo if found is not None and found.answer is not None else "(sem resumo)"
            return short_text(text, SHORTFALL_SUMMARY_WORDS)

        def where(clip_id: str) -> str:
            return f" (em fora: {short_text(out_reason[clip_id], 6)})" if clip_id in out_reason else ""

        unused = [
            f"- {c} · {round(material.speech_sec(c))} s de fala{where(c)} — {summary(c)}"
            for c in material.clip_ids if material.has_speech(c) and c not in in_blocks
        ]
        library = [
            f"- {c} · {clock(material.clips[c].duration_sec)}{where(c)} — {summary(c)}"
            for c in material.clip_ids if not material.has_speech(c) and c not in in_blocks
        ]
        if not unused and not library:
            return None
        return StructureShortfall(
            target_sec=target, speech_sec=speech, unused_lines=unused,
            library_lines=library,
        )

    # --- E11: passada 2 de um bloco ---

    def _block(self, block: Block, structure: StructureRun, usage: Usage, first_block: bool = False) -> DetailRun:
        """2a por clipe → trechos → 2b → 2c; a faixa principal do bloco montada pelo código. Atualiza `usage`.

        `first_block`: nenhum bloco anterior pôs nada no vídeo (o vídeo começa aqui: sem respiro na posição 0,
        a não ser que o pedido peça abertura com paisagem)."""
        material = self.material
        header = BlockHeader(
            block_id=block.id, title=block.title, subject=block.subject, context=self.context,
            target_total_sec=structure.target_total_sec, block_target_sec=block.target_sec,
            used_before_sec=round(usage.used_sec, 3),
        )
        detail = DetailRun(
            unit_id=block.id, block_id=block.id, clip_ids=list(block.clip_ids), origin="ia",
            used_before_sec=round(usage.used_sec, 3), target_sec=block.target_sec,
        )
        free = {
            c: [s.id for s in material.clip_sentences.get(c, []) if s.id not in usage.sentences] for c in block.clip_ids
        }
        speaking = [c for c in block.clip_ids if free[c]]
        whole = sentences_sec(material, [s for c in speaking for s in free[c]])
        detail.available_sec = round(whole, 3)

        # 2a: cortes, um pedido por clipe com fala (clipe grande por partes).
        kept: list[str] = []
        removed: list[str] = []
        fixed = {c: reason for c in speaking if (reason := self._no_cuts_reason(free[c]))}
        targets = _clip_targets(block.target_sec, {c: sentences_sec(material, free[c]) for c in speaking}, set(fixed))
        asked = len(speaking) - len(fixed)
        for clip_id in speaking:
            runs_before = len(detail.cut_runs)
            clip_sec = sentences_sec(material, free[clip_id])
            target = targets.get(clip_id)
            if clip_id in fixed:  # sem o que cortar: entra inteiro (a estrutura já decidiu que entra)
                self.calls_done += 1
                detail.cut_runs.append(CutRun(
                    unit_id=f"{block.id}.{clip_id}", clip_id=clip_id, origin="ia", available_sec=round(clip_sec, 3),
                    target_sec=target, result_sec=round(clip_sec, 3), kept_ids=list(free[clip_id]),
                    no_question=fixed[clip_id],
                ))
                kept += free[clip_id]
                continue
            parts = self._cut_parts(header, clip_id, free[clip_id], target, asked)
            for k, (moment_ids, ids) in enumerate(parts, start=1):
                part_target = target
                if target is not None and len(parts) > 1:
                    part_target = round(target * sentences_sec(material, ids) / clip_sec, 1) if clip_sec > 0 else None
                unit_id = f"{block.id}.{clip_id}" + (_suffix(k - 1) if len(parts) > 1 else "")
                cut_run = self._cuts(header, unit_id, clip_id, moment_ids, ids, part_target, k, len(parts), asked)
                detail.cut_runs.append(cut_run)
                kept += cut_run.kept_ids
            if all(r.removed for r in detail.cut_runs[runs_before:]):
                removed.append(clip_id)
                if len(parts) > 1:  # um clipe só já tem o aviso do próprio pedido (com o motivo)
                    detail.warnings.append(f"IA tirou o clipe {clip_id} (todas as partes)")
        if removed:
            detail.removed_clip_ids = removed
            detail.clip_ids = [c for c in block.clip_ids if c not in removed]
        take_sentences(usage, kept, material)
        trechos = build_trechos(detail.clip_ids, set(kept), material)
        detail.trechos = trechos
        detail.speech_sec = round(sum(t.seconds for t in trechos), 3)

        # 2b: inclusões e respiros; 2c: coberturas.
        detail.inserts = self._inserts(header, block, trechos, structure, usage, detail, first_block)
        detail.coverage = self._coverage(header, detail.clip_ids, trechos, structure, usage)

        by_after: dict[int, list] = {}
        for choice in detail.inserts.inserts:
            by_after.setdefault(choice.after_trecho, []).append(choice)

        def emit(after: int) -> None:
            for choice in by_after.get(after, []):
                moment = material.moments[choice.moment_id]
                detail.keep.append(KeepRange(
                    kind="momento", clip_id=moment.clip_id, moment_id=moment.id, duration_sec=choice.duration_sec,
                    start_sec=choice.start_sec, breath=choice.breath,
                ))

        emit(0)
        for trecho in trechos:
            detail.keep.append(KeepRange(
                kind="frases", clip_id=trecho.clip_id, from_sentence=trecho.sentence_ids[0],
                to_sentence=trecho.sentence_ids[-1], sentence_ids=list(trecho.sentence_ids),
            ))
            emit(trecho.number)
        detail.coverages = list(detail.coverage.coverages)
        inserted = sum(
            min(c.duration_sec or material.moments[c.moment_id].duration, material.moments[c.moment_id].duration)
            for c in detail.inserts.inserts
        )
        detail.result_sec = round(detail.speech_sec + inserted, 3)
        detail.breath_sec = round(sum(c.duration_sec or 0.0 for c in detail.inserts.inserts if c.breath), 3)
        usage.used_sec += detail.result_sec
        detail.library_offered = [c.moment_id for c in detail.inserts.candidates if c.source == "acervo"]
        if not detail.keep:
            detail.warnings.append("nada ficou neste bloco (sem fala livre e sem imagens)")

        origins = [r.origin for r in detail.cut_runs if not r.no_question]
        origins += [s.origin for s in (detail.inserts, detail.coverage) if s.requests]
        if "fallback" in origins:
            detail.origin = "fallback"
        elif origins and all(o == "cache" for o in origins):
            detail.origin = "cache"
        for cut_run in detail.cut_runs:
            detail.warnings += [f"{cut_run.unit_id}: {w}" for w in cut_run.warnings]
        for name, sub in (("inclusões", detail.inserts), ("coberturas", detail.coverage)):
            detail.warnings += [f"{name}: {w}" for w in sub.warnings]
        return detail

    # --- 2a ---

    def _cut_parts(
        self, header: BlockHeader, clip_id: str, ids: list[str], target: Optional[float], block_clips: int
    ) -> list[tuple[Optional[list[str]], list[str]]]:
        """[(None, frases)] se o clipe inteiro cabe num pedido; senão (momentos, frases) de cada parte."""
        material = self.material
        available = self.budget.cuts_available

        def prompt(moment_ids: Optional[list[str]], part_ids: list[str], parts: int) -> str:
            return build_cuts_prompt(CutsPromptInput(
                header=header, clip_id=clip_id, section=material.section(clip_id, moment_ids),
                available_sec=sentences_sec(material, part_ids), target_sec=target, part=1, parts=parts,
                block_clips=block_clips, continues=continuing_sentences(part_ids, material),
            ))

        if tokens(prompt(None, ids, 1)) <= available:
            return [(None, ids)]
        wanted = set(ids)

        def sentences_of(moment_ids: list[str]) -> list[str]:
            return [s.id for s in material.moment_sentences(moment_ids) if s.id in wanted]

        chunks: list[list[str]] = []
        current: list[str] = []
        for moment in material.clip_moments.get(clip_id, []):
            trial = current + [moment.id]
            if current and tokens(prompt(trial, sentences_of(trial), 2)) > available:
                chunks.append(current)
                trial = [moment.id]
            current = trial
        if current:
            chunks.append(current)
        parts = [(chunk, sentences_of(chunk)) for chunk in chunks]
        parts = [p for p in parts if p[1]]
        placed = {s for _, part_ids in parts for s in part_ids}
        leftover = [s for s in ids if s not in placed]  # frases sem momento: vão com a última parte
        if not parts:
            return [(None, ids)]
        if leftover:
            moment_ids, part_ids = parts[-1]
            parts[-1] = (moment_ids, [s for s in ids if s in set(part_ids) | set(leftover)])
        return parts

    def _no_cuts_reason(self, ids: list[str]) -> str:
        """Por que o clipe não tem pergunta de cortes ("" = tem): 1 frase ou fala até `CUT_MIN_SPEECH_SEC`."""
        if len(ids) == 1:
            return "1 frase"
        if sentences_sec(self.material, ids) <= CUT_MIN_SPEECH_SEC:
            return "fala curta"
        return ""

    def _duration_problem(self, check, target: Optional[float]) -> Optional[str]:
        """Resultado fora da meta ± `DURATION_TOLERANCE` (só informação; None = dentro ou sem meta).

        Abaixo da meta sem nada cortado, não há o que ajustar: não pede de novo.
        """
        if target is None or target <= 0:
            return None
        low, high = target * (1 - DURATION_TOLERANCE), target * (1 + DURATION_TOLERANCE)
        if low <= check.result_sec <= high:
            return None
        if check.result_sec < low and not check.cuts:
            return None
        return duration_retry_line(check.result_sec, target, low, high)

    def _cut_list(self, check, target: Optional[float]) -> list[str]:
        """Fora da meta: as frases com os segundos para a nova tentativa (acima: as que ficaram, para escolher o
        que cortar; abaixo: as cortadas, para escolher o que devolver)."""
        if target is None:
            return []
        above = check.result_sec > target
        material = self.material
        ids = check.kept if above else [c.sentence_id for c in check.cuts]
        items = [(sid, sentences_sec(material, [sid]), material.sentences[sid].text) for sid in ids]
        return sentence_retry_lines(above, items)

    def _cuts(
        self, header: BlockHeader, unit_id: str, clip_id: str, moment_ids: Optional[list[str]], ids: list[str],
        target: Optional[float], part: int, parts: int, block_clips: int,
    ) -> CutRun:
        material = self.material
        self._progress(f"cortes {unit_id}")
        data = CutsPromptInput(
            header=header, clip_id=clip_id, section=material.section(clip_id, moment_ids),
            available_sec=round(sentences_sec(material, ids), 3), target_sec=target, part=part, parts=parts,
            block_clips=block_clips, continues=continuing_sentences(ids, material),
        )
        run = CutRun(
            unit_id=unit_id, clip_id=clip_id, moment_ids=moment_ids, origin="ia", available_sec=data.available_sec,
            target_sec=target,
        )
        decision = self._decide(
            "cortes", CUTS_PROMPT_VERSION, unit_id, CutsAnswer,
            build=lambda errors: build_cuts_prompt(data, errors),
            make_request=lambda prompt: CutsRequest(
                unit_id=unit_id, prompt=prompt, material=material, clip_id=clip_id, sentence_ids=list(ids),
                target_sec=target,
            ),
            call=self.ai.cuts, check_of=lambda answer: validate_cuts(answer, ids, material), stats=run.stats,
            extra_problem=lambda check: None if check.removed else self._duration_problem(check, target),
            retry_details=lambda check: self._cut_list(check, target),
            choose=lambda checks: _closest_cut(checks, target),
        )
        self.calls_done += 1
        run.estimated_tokens, run.attempts = decision.estimated, decision.attempts
        run.tries = [
            CutAttempt(attempt=n, result_sec=c.result_sec, all_cut=not c.out and not c.kept, out=c.out)
            for n, c in decision.tries
        ]
        run.chosen_attempt = decision.chosen
        if decision.estimated > self.budget.cuts_available:
            run.warnings.append(f"o pedido (~{decision.estimated} tokens) passa do disponível ({self.budget.cuts_available}) mesmo dividido")
        check = decision.check
        if check is None:
            check = validate_cuts(CutsAnswer(), ids, material)
            run.origin = "fallback"
            message = (
                f"CORTES {unit_id} PELO FALLBACK NEUTRO (a IA falhou de vez): nada cortado. "
                f"Motivo: {decision.attempts[-1] if decision.attempts else 'sem resposta'}"
            )
            run.warnings.append(message)
            self._notice(message)
        else:
            run.origin = decision.origin  # type: ignore[assignment]
            run.notes = decision.answer.notas
            run.normalized_ids = check.normalized
            if decision.attempts and check.discarded:
                run.warnings.append(f"aceitos os itens válidos; {len(check.discarded)} item(ns) inválido(s) descartado(s)")
            if decision.problem:
                run.duration_off = True
                run.warnings.append(f"fora da meta depois da nova tentativa, aceito assim ({decision.problem})")
            if check.removed and not check.out:  # só listas que cortam tudo: tirar o clipe não é da 2a
                check = validate_cuts(CutsAnswer(), ids, material)
                run.kept_whole = True
                run.warnings.append(ALL_CUT_KEPT)
            elif check.removed:
                run.removed = True
                if parts > 1:
                    run.warnings.append(f"IA tirou a parte {unit_id} do clipe {clip_id} (fora: true)")
                else:
                    run.warnings.append(f"IA tirou o clipe {clip_id} (fora: true)")
        if run.removed:
            check.kept, check.result_sec = [], 0.0
        run.cuts, run.kept_ids, run.result_sec = check.cuts, check.kept, check.result_sec
        run.discarded = check.discarded
        run.joined = list(check.joined)
        run.warnings += [f"cortar descartado ({d.item}): {d.reason}" for d in check.discarded]
        return run

    # --- 2b ---

    def _free_moment(self, moment: Moment, usage: Usage) -> bool:
        return (
            moment.duration > 0 and moment.id not in usage.moments
            and not usage.conflict(moment.clip_id, moment.start_sec, moment.end_sec)
        )

    def _inserts(
        self, header: BlockHeader, block: Block, trechos: list[Trecho], structure: StructureRun, usage: Usage,
        detail: DetailRun, first_block: bool = False,
    ) -> SubpassRun:
        """2b do bloco: mede a sobra, as posições e os candidatos (informação); a IA escolhe, e "nenhum" vale."""
        material = self.material
        sub = SubpassRun(origin="ia")
        spare = round(block.target_sec - detail.speech_sec, 3) if block.target_sec else None
        sub.spare_sec = spare
        has_time = spare is not None and spare > 0
        if trechos and not has_time:
            sub.no_question = "sem sobra para respiro" if spare is not None else "sem alvo do bloco: sem sobra medida"
            self.calls_done += 1
            return sub
        positions = insert_positions(trechos, material, first_block, asks_landscape_opening(self.context.request))
        breath_positions = positions.breath if has_time else []
        sub.breath_positions, sub.own_positions = list(breath_positions), list(positions.own)
        sub.skipped = list(positions.refused)
        if not has_time and positions.breath:
            sub.skipped += [Skipped(number=p, reason="sem sobra para respiro") for p in positions.breath]

        own = [
            m for c in detail.clip_ids for m in material.clip_moments.get(c, [])
            if not m.sentence_ids and self._free_moment(m, usage)
        ]
        if trechos:
            own = own[:INSERT_CANDIDATES]
        library: list[tuple[Moment, list[str]]] = []
        if breath_positions and (trechos or own):
            pool = [
                m for c in structure.library_clip_ids for m in material.clip_moments.get(c, [])
                if not m.sentence_ids and self._free_moment(m, usage)
            ]
            neighbors: list[Trecho] = []
            for p in breath_positions:
                neighbors += [t for t in trechos if t.number in (p, p + 1) and t not in neighbors]
            text = " ".join(material.sentences[s].text for t in neighbors for s in t.sentence_ids)
            places = [trecho_place(t, material, end) for t in neighbors for end in (False, True)]
            places = [p for p in places if p]
            related, rest = [], []
            for moment in library_ranking(material, detail.clip_ids, pool):
                found = breath_affinity(moment, text, places)
                (related if found else rest).append((moment, found))
            library = (related + rest)[: max(0, INSERT_CANDIDATES - len(own))]
        trecho_lines = [_trecho_line(t, material, TRECHO_WORDS) for t in trechos]

        def make(moments: list[tuple[Moment, str, list[str]]]) -> list[Candidate]:
            return [
                Candidate(
                    letter=letter, moment_id=m.id, clip_id=m.clip_id, source=source, words=found,
                    duration_sec=round(m.duration if source == "bloco" else min(m.duration, RESPIRO_MAX_SEC), 3),
                    description=short_text(m.description, CANDIDATE_WORDS) if m.description else "", place=m.place,
                )
                for letter, (m, source, found) in zip(letters(len(moments)), moments)
            ]

        def data_for(candidates: list[Candidate], left: int) -> InsertsPromptInput:
            return InsertsPromptInput(
                header=header, trecho_lines=trecho_lines, candidate_lines=[_insert_line(c, material) for c in candidates],
                current_sec=detail.speech_sec, library_left=left, spare_sec=spare, own_positions=positions.own,
                breath_position_lines=[f"{p} ({positions.notes[p]})" for p in breath_positions],
                breath_max_sec=RESPIRO_MAX_SEC,
            )

        chosen = [(m, "bloco", []) for m in own]
        for n, (moment, found) in enumerate(library):
            trial = make(chosen + [(moment, "acervo", found)])
            if tokens(build_inserts_prompt(data_for(trial, len(library) - n - 1))) > self.budget.inserts_available:
                break
            chosen.append((moment, "acervo", found))
        candidates = make(chosen)
        sub.candidates = candidates
        left = len(own) + len(library) - len(candidates)
        if left:
            sub.warnings.append(f"acervo: {len(candidates) - len(own)} de {len(library)} imagens couberam no pedido")
        if not candidates:
            if not trechos:
                sub.warnings.append("bloco sem fala e sem imagens livres")
            sub.no_question = (
                "nenhuma posição de respiro e nenhuma imagem do bloco" if not breath_positions and not own
                else "nenhuma imagem livre para oferecer"
            )
            self.calls_done += 1
            return sub
        data = data_for(candidates, left)
        limit = spare if spare is not None else None
        self._progress(f"inclusões {block.id}")

        def check_of(answer):
            return validate_inserts(
                answer, trechos, candidates, material, usage, spare_sec=limit,
                breath_positions=breath_positions, own_positions=positions.own,
            )

        decision = self._decide(
            "insercoes", INSERTS_PROMPT_VERSION, block.id, InsertsAnswer,
            build=lambda errors: build_inserts_prompt(data, errors),
            make_request=lambda prompt: InsertsRequest(
                unit_id=block.id, prompt=prompt, material=material, trechos=trechos, candidates=candidates,
                block_target_sec=block.target_sec, current_sec=detail.speech_sec, spare_sec=spare,
                breath_positions=list(breath_positions), own_positions=list(positions.own),
            ),
            call=self.ai.inserts, check_of=check_of, stats=sub.stats,
        )
        self.calls_done += 1
        sub.requests, sub.estimated_tokens, sub.attempts = 1, decision.estimated, decision.attempts
        check = decision.check
        if check is None:
            check = check_of(neutral_inserts(trechos, candidates))
            sub.origin = "fallback"
            message = (
                f"INCLUSÕES {block.id} PELO FALLBACK NEUTRO (a IA falhou de vez): "
                + ("nenhuma imagem a mais. " if trechos else "todas as imagens do bloco, em ordem. ")
                + f"Motivo: {decision.attempts[-1] if decision.attempts else 'sem resposta'}"
            )
            sub.warnings.append(message)
            self._notice(message)
        else:
            sub.origin = decision.origin  # type: ignore[assignment]
            sub.notes = decision.answer.notas
        sub.inserts, sub.discarded = check.inserts, check.discarded
        sub.warnings += check.warnings + [f"inserir descartado ({d.item}): {d.reason}" for d in check.discarded]
        _replace_usage(usage, check.usage)
        return sub


    # --- 2c ---

    def _coverage(
        self, header: BlockHeader, clip_ids: list[str], trechos: list[Trecho], structure: StructureRun, usage: Usage
    ) -> SubpassRun:
        """Só trechos ditos para a câmera têm pergunta: nos outros, a própria imagem já mostra o assunto."""
        material = self.material
        sub = SubpassRun(origin="ia")
        pool_clips = [*clip_ids, *structure.library_clip_ids]
        pool = [m for c in pool_clips for m in material.clip_moments.get(c, []) if self._free_moment(m, usage)]
        pool = library_ranking(material, clip_ids, pool)
        letter_of: dict[str, str] = {}
        questions: dict[int, list[Candidate]] = {}
        texts: dict[int, CoverageQuestion] = {}
        for trecho in trechos:
            talking = talking_sentences(trecho, material)
            if not talking:
                sub.skipped.append(Skipped(number=trecho.number, reason=NOT_TALKING))
                continue
            spoken = trecho.model_copy(update={"sentence_ids": talking})
            speech = " ".join(material.sentences[s].text for s in talking)
            options = [m for m in pool if m.clip_id != trecho.clip_id]
            matches = coverage_matches(
                speech, speech_places(spoken, material), options, COVERAGE_CANDIDATES, COVERAGE_SAYS, COVERAGE_PLACE,
            )
            if not matches:
                continue
            candidates = []
            for match in matches:
                letter = letter_of.setdefault(match.moment.id, letters(len(letter_of) + 1)[-1])
                candidates.append(Candidate(
                    letter=letter, moment_id=match.moment.id, clip_id=match.moment.clip_id,
                    duration_sec=round(match.moment.duration, 3),
                    source="acervo" if match.moment.clip_id in structure.library_clip_ids else "outro clipe",
                    kind=match.kind, words=match.words,
                ))
            questions[trecho.number] = candidates
            texts[trecho.number] = CoverageQuestion(
                trecho_line=_coverage_trecho_line(trecho, material),
                candidate_lines=[_coverage_line(c, material) for c in candidates],
            )
        unique: dict[str, Candidate] = {}
        for candidates in questions.values():
            for candidate in candidates:
                unique.setdefault(candidate.letter, candidate)
        sub.candidates = list(unique.values())
        if not questions:
            if trechos and len(sub.skipped) == len(trechos):
                sub.no_question = "nenhum trecho dito para a câmera"
            self.calls_done += 1
            return sub

        # Os trechos perguntados em pedidos que cabem no contexto (em ordem).
        groups: list[list[int]] = []
        current: list[int] = []
        for number in questions:
            trial = current + [number]
            prompt = build_coverage_prompt(CoveragePromptInput(header=header, questions=[texts[n] for n in trial], parts=2))
            if current and tokens(prompt) > self.budget.coverage_available:
                groups.append(current)
                trial = [number]
            current = trial
        groups.append(current)

        origins = []
        for k, group in enumerate(groups, start=1):
            unit = header.block_id + (_suffix(k - 1) if len(groups) > 1 else "")
            asked = {n: questions[n] for n in group}
            data = CoveragePromptInput(header=header, questions=[texts[n] for n in group], part=k, parts=len(groups))
            self._progress(f"coberturas {unit}")
            decision = self._decide(
                "coberturas", COVERAGE_PROMPT_VERSION, unit, CoveragesAnswer,
                build=lambda errors, data=data: build_coverage_prompt(data, errors),
                make_request=lambda prompt, asked=asked: CoverageRequest(
                    unit_id=unit, prompt=prompt, material=material, trechos=trechos, questions=asked,
                ),
                call=self.ai.coverages,
                check_of=lambda answer, asked=asked: validate_coverages(answer, asked, trechos, material, usage),
                stats=sub.stats,
            )
            sub.requests += 1
            sub.estimated_tokens += decision.estimated
            sub.attempts += decision.attempts
            check = decision.check
            if check is None:
                origins.append("fallback")
                message = (
                    f"COBERTURAS {unit} PELO FALLBACK NEUTRO (a IA falhou de vez): sem cobertura. "
                    f"Motivo: {decision.attempts[-1] if decision.attempts else 'sem resposta'}"
                )
                sub.warnings.append(message)
                self._notice(message)
                continue
            origins.append(decision.origin)
            sub.notes = " · ".join(n for n in (sub.notes, decision.answer.notas) if n)
            sub.coverages += check.coverages
            sub.nones += check.nones
            sub.discarded += check.discarded
            sub.warnings += [f"cobertura descartada ({d.item}): {d.reason}" for d in check.discarded]
            _replace_usage(usage, check.usage)
        self.calls_done += 1
        sub.origin = "fallback" if "fallback" in origins else ("cache" if all(o == "cache" for o in origins) else "ia")
        return sub


def _closest_cut(checks: list, target: Optional[float]) -> int:
    """A resposta da 2a usada (validação, não decisão: todas são da IA): um fora: true explícito (o último)
    vale; senão, entre as que mantêm alguma frase, a mais perto da meta (empate: a primeira; sem meta, a
    última). Só as que cortam tudo: a última (o clipe entra inteiro, com aviso)."""
    outs = [k for k, c in enumerate(checks) if c.out]
    if outs:
        return outs[-1]
    keeping = [k for k, c in enumerate(checks) if c.kept]
    if not keeping:
        return len(checks) - 1
    if target is None:
        return keeping[-1]
    return min(keeping, key=lambda k: (abs(checks[k].result_sec - target), k))


def _clip_targets(block_target: Optional[float], speech: dict[str, float], fixed: set[str]) -> dict[str, float]:
    """Meta de cada clipe com fala do bloco (aritmética, sem decisão): os sem pergunta (`fixed`) entram com a
    fala inteira; o resto da meta do bloco é repartido pela fala de cada clipe, e nenhuma meta passa da fala do
    clipe (o excesso vai para os outros, na mesma proporção). Sem meta do bloco: {} (sem meta)."""
    if not block_target or sum(speech.values()) <= 0:
        return {}
    targets = {c: speech[c] for c in fixed}
    remaining = block_target - sum(targets.values())
    open_ = [c for c in speech if c not in fixed]
    while open_:
        total = sum(speech[c] for c in open_)
        share = {c: max(remaining, 0.0) * speech[c] / total if total > 0 else 0.0 for c in open_}
        over = [c for c in open_ if share[c] > speech[c]]
        if not over:
            targets.update(share)
            break
        for c in over:
            targets[c] = speech[c]
            remaining -= speech[c]
            open_.remove(c)
    return {c: round(v, 1) for c, v in targets.items()}


def _block_content_sec(block: Block, material: EditorMaterial) -> float:
    """Fala dos clipes do bloco (sem fala: a duração dos clipes)."""
    speech = sum(material.speech_sec(c) for c in block.clip_ids)
    return speech if speech > 0 else sum(material.clips[c].duration_sec for c in block.clip_ids)


def _rescale_blocks(blocks: list[Block], total: Optional[float], material: EditorMaterial) -> None:
    """Metas dos blocos reescaladas para somar o alvo, na proporção que a IA deu (bloco sem meta pesa o
    conteúdo dele). Sem alvo ou sem nenhuma meta da IA, nada muda. A meta da IA fica em `ai_target_sec`."""
    if not total or total <= 0 or not any(b.target_sec for b in blocks):
        return
    weights = {b.id: b.target_sec if b.target_sec and b.target_sec > 0 else _block_content_sec(b, material) for b in blocks}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        return
    for block in blocks:
        block.ai_target_sec = block.target_sec
        block.target_sec = round(total * weights[block.id] / weight_sum, 1)


def _suffix(k: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return alphabet[k] if k < len(alphabet) else str(k + 1)


def _replace_usage(usage: Usage, new: Usage) -> None:
    """O uso do serviço passa a ser o da conferência aceita (os validadores trabalham sobre uma cópia)."""
    usage.__dict__.update(new.__dict__)


def _issues(moment: Optional[Moment]) -> str:
    return f" [{', '.join(moment.issues)}]" if moment is not None and moment.issues else ""


def _trecho_line(trecho: Trecho, material: EditorMaterial, words: int) -> str:
    """`3: S012–S015 (P006, 6 s): texto…`"""
    first, last = trecho.sentence_ids[0], trecho.sentence_ids[-1]
    ids = first if first == last else f"{first}–{last}"
    text = " ".join(material.sentences[s].text for s in trecho.sentence_ids)
    return f"{trecho.number}: {ids} ({trecho.clip_id}, {round(trecho.seconds)} s): {short_text(text, words)}"


def _insert_line(candidate: Candidate, material: EditorMaterial) -> str:
    """`- A: descrição curta · lugar: lago · 6,0s · B004 (acervo; em comum: lago) [problemas]` (respiro: a duração
    que entra, no máximo `RESPIRO_MAX_SEC`)."""
    moment = material.moments[candidate.moment_id]
    parts = [candidate.description or "(sem descrição)"]
    if candidate.place:
        parts.append(f"lugar: {candidate.place}")
    parts.append(_fmt_sec(candidate.duration_sec))
    where = "do bloco" if candidate.source == "bloco" else "acervo"
    if candidate.words:
        where += f"; em comum: {', '.join(candidate.words)}"
    parts.append(f"{moment.clip_id} ({where})")
    return f"- {candidate.letter}: {' · '.join(parts)}{_issues(moment)}"


def _coverage_trecho_line(trecho: Trecho, material: EditorMaterial) -> str:
    """`Trecho 3 (S012–S015 · P006 · 6 s; imagem de quem fala: … [problemas]): "texto"`"""
    first, last = trecho.sentence_ids[0], trecho.sentence_ids[-1]
    ids = first if first == last else f"{first}–{last}"
    sentence = material.sentences[first]
    moment = material.moments.get(sentence.moment_ids[0]) if sentence.moment_ids else None
    image = ""
    if moment is not None:
        image = f"; imagem de quem fala: {moment.description or '(sem descrição)'}{_issues(moment)}"
        if moment.talking_to_camera:
            image += ", fala para a câmera"
    text = " ".join(material.sentences[s].text for s in trecho.sentence_ids)
    return (
        f"Trecho {trecho.number} ({ids} · {trecho.clip_id} · {round(trecho.seconds)} s{image}): "
        f"\"{short_text(text, COVERAGE_TRECHO_WORDS)}\""
    )


def _coverage_line(candidate: Candidate, material: EditorMaterial) -> str:
    """`C: descrição · lugar: lago · 9,0s · D001 [problemas] — mostra o que é dito; em comum: lago`"""
    moment = material.moments[candidate.moment_id]
    parts = [moment.description or "(sem descrição)"]
    if moment.place:
        parts.append(f"lugar: {moment.place}")
    parts += [_fmt_sec(moment.duration), moment.clip_id]
    return (
        f"{candidate.letter}: {' · '.join(parts)}{_issues(moment)} — {candidate.kind}; "
        f"em comum: {', '.join(candidate.words)}"
    )


def _join_summaries(answers: list[ClipSummaryAnswer]) -> ClipSummaryAnswer:
    """Clipe resumido em pedaços: os resumos em sequência, os primeiros 3 destaques e problemas, o 1º tom e
    "fala para a câmera" se algum pedaço tem (junção do código, sem nova chamada)."""
    if len(answers) == 1:
        return answers[0]
    return ClipSummaryAnswer(
        resumo=" ".join(a.resumo for a in answers),
        destaques=[n for a in answers for n in a.destaques],
        problemas=[n for a in answers for n in a.problemas],
        tom=next((a.tom for a in answers if a.tom), ""),
        tem_fala_para_camera=any(a.tem_fala_para_camera for a in answers),
    )
