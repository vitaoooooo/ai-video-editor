"""Passadas do Editor de IA: resumos por clipe (E9), estrutura (E10) e detalhe por bloco (E11).

Cada chamada aceita vai para um cache em disco assim que termina (um Ctrl+C no meio não perde o que já
foi feito), em `<projects>/_cache/editor_ia/<passada>/<fingerprint>.json`, com fingerprint = o pedido
exato (sem a lista de erros de uma nova tentativa) + versão do pedido da passada + provedor, cadeia de
modelos e raciocínio. Como o pedido do resumo só contém a seção do clipe no roteiro, mudar o pedido de
outra passada não refaz os resumos. Respostas inaproveitáveis não vão para o cache (a próxima execução pede
de novo).

Nova tentativa (uma, com a lista de erros): estrutura sem bloco válido ou com clipe com fala esquecido;
detalhe com metade ou mais dos itens inválida, sem nada no bloco ou com a duração resultante fora do alvo
do pedaço ± `DURATION_TOLERANCE` (a nova tentativa mostra os números). Depois dela, a última resposta
aproveitável é aceita com os itens válidos (os inválidos saem com aviso; duração fora do alvo é registrada);
clipe com fala ainda esquecido vai para um bloco próprio. Fallback neutro (`validate.neutral_*`, com aviso)
só sem resposta aproveitável (erro da API, fora do formato, nenhum item válido, nada no bloco). Depois de
`MAX_CONSECUTIVE_FAILURES` erros seguidos da API (não de validação), a IA não é mais chamada nesta execução
e o resto vai pelos fallbacks. Erro de autenticação interrompe tudo.
"""

import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from ai_video_editor.editor.budget import LIBRARY_MIN_TOKENS, Budget, tokens
from ai_video_editor.editor.ids import IdResolver
from ai_video_editor.editor.material import EditorMaterial, library_ranking
from ai_video_editor.editor.prompt import (
    DETAIL_PROMPT_VERSION,
    STRUCTURE_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    DetailPromptInput,
    EditorContext,
    StructureClip,
    build_detail_prompt,
    build_structure_prompt,
    build_summary_prompt,
    duration_retry_line,
)
from ai_video_editor.editor.provider import DetailRequest, EditorAI, StructureRequest, SummaryRequest
from ai_video_editor.editor.schema import (
    Block,
    BlockAnswer,
    CallStats,
    ClipSummary,
    ClipSummaryAnswer,
    DetailRun,
    EditorRun,
    StructureAnswer,
    StructureRun,
    SummaryNote,
)
from ai_video_editor.editor.validate import (
    Unit,
    Usage,
    neutral_detail,
    neutral_structure,
    place_forgotten,
    sentences_sec,
    validate_detail,
    validate_structure,
)
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIResponseValidationError,
)
from ai_video_editor.pipeline.project import fingerprint
from ai_video_editor.pipeline.schema import StageProgress

EDITOR_CACHE_DIR = Path("_cache") / "editor_ia"
# Tentativas por pedido: a 1ª + uma nova com a lista de erros.
MAX_ATTEMPTS = 2
# Erros seguidos da API (não de validação) antes de parar de chamar a IA nesta execução.
MAX_CONSECUTIVE_FAILURES = 3
# Frases do bloco anterior que vão como continuidade.
CONTINUITY_SENTENCES = 2
# Faixa aceita da duração resultante de um pedaço: alvo ± 35%. Fora dela, UMA nova tentativa com os números;
# se ainda ficar fora, aceita e registra (a duração é informação para a IA, não uma decisão do código).
DURATION_TOLERANCE = 0.35

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


class EditorService:
    """Executa as três passadas sobre o material e devolve o `EditorRun` (sem tempos de mídia: isso é a montagem)."""

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

    def _ask(self, call: Callable, request, stats: CallStats) -> tuple[Optional[BaseModel], Optional[str]]:
        """(resposta, erro). Erro de validação conta como tentativa; erro da API, como falha seguida."""
        if self.gave_up:
            return None, f"a IA não foi chamada ({MAX_CONSECUTIVE_FAILURES} erros seguidos da API antes)"
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
            stats.seconds = round(stats.seconds + time.perf_counter() - started, 3)
        return answer, error

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
        units = [(block, self._units(block)) for block in run.structure.blocks]
        self.calls_total = self.calls_done + sum(len(u) for _, u in units)
        for block, block_units in units:
            if len(block_units) > 1:
                note = f"bloco {block.id} dividido em {len(block_units)} pedaços para caber no contexto ({self.budget.num_ctx} tokens)"
                run.warnings.append(note)
                self._notice(note)
            for unit in block_units:
                detail, usage = self._detail(unit, run.structure, usage)
                run.details.append(detail)
                run.detail_stats.add(detail.stats)
                run.warnings += [f"{detail.unit_id}: {w}" for w in detail.warnings]
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
            self._progress(f"resumo {clip_id}" + (f" ({k}/{len(parts)})" if len(parts) > 1 else ""))
            section = material.section(clip_id, moment_ids)
            prompt = build_summary_prompt(section, k, len(parts))
            key = self.key("resumos", SUMMARY_PROMPT_VERSION, prompt)
            answer = self.cache.load("resumos", key, ClipSummaryAnswer)
            if answer is not None:
                summary.cached_parts += 1
                stats.cached += 1
            else:
                errors: list[str] = []
                for _ in range(MAX_ATTEMPTS):
                    request = SummaryRequest(
                        clip_id=clip_id, prompt=build_summary_prompt(section, k, len(parts), errors),
                        material=material, moment_ids=moment_ids, part=k, parts=len(parts),
                    )
                    answer, error = self._ask(self.ai.summarize, request, stats)
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

    def _structure(self, summaries: dict[str, ClipSummary]) -> StructureRun:
        material, context = self.material, self.context
        clips = [
            StructureClip(c, material.clip_line(c), summaries.get(c), tokens(material.section(c) + "\n"))
            for c in material.clip_ids
        ]
        block_tokens = max(0, self._content_budget(Block(id="B01", clip_ids=[])))
        result = StructureRun(origin="ia")

        def prompt_for(errors: Optional[list[str]] = None) -> str:
            return build_structure_prompt(context, clips, result.prompt_mode == "compacto", errors, block_tokens)

        prompt = prompt_for()
        if tokens(prompt) > self.budget.structure_available:
            result.prompt_mode = "compacto"
            prompt = prompt_for()
            result.warnings.append(
                "a lista de clipes com os resumos completos não coube no contexto; a estrutura recebeu os resumos encurtados"
            )
        self._progress("estrutura")
        answer: Optional[StructureAnswer] = None
        check = None
        if tokens(prompt) > self.budget.structure_available:
            result.attempts.append(f"o pedido da estrutura (~{tokens(prompt)} tokens) não cabe no contexto ({self.budget.num_ctx})")
        else:
            key = self.key("estrutura", STRUCTURE_PROMPT_VERSION, prompt)
            cached = self.cache.load("estrutura", key, StructureAnswer)
            if cached is not None:
                check = validate_structure(cached, material)
                if check.blocks:
                    answer, result.origin = cached, "cache"
                    result.stats.cached += 1
            errors: list[str] = []
            for _ in range(0 if answer is not None else MAX_ATTEMPTS):
                request = StructureRequest(prompt=prompt_for(errors), material=material, context=context, summaries=summaries)
                raw, error = self._ask(self.ai.structure, request, result.stats)
                if raw is not None:
                    trial = validate_structure(raw, material)
                    if trial.blocks:  # aproveitável: a última aproveitável vale se a nova tentativa não resolver
                        answer, check = raw, trial
                    if trial.refused is None:
                        break
                    error = f"{trial.refused}: " + "; ".join(trial.errors[:12])
                errors.append(error)
                result.attempts.append(error)
                if self.gave_up:
                    break
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
        if check.forgotten:
            result.blocks = place_forgotten(check.blocks, check.forgotten, material)
            message = (
                f"estrutura: clipe(s) com fala esquecido(s) pela IA ({', '.join(check.forgotten)}) entraram em bloco "
                "próprio, na ordem do catálogo"
            )
            result.warnings.append(message)
            self._notice(message)
        if check.discarded and result.origin != "fallback":
            result.warnings.append(f"estrutura: {len(check.discarded)} item(ns) descartado(s) na validação")
        placed = {c for b in result.blocks for c in b.clip_ids} | {o.clipe for o in result.out}
        result.library_clip_ids = [c for c in material.clip_ids if c not in placed]
        if context.target_sec is not None:
            result.target_total_sec, result.target_origin = context.target_sec, "pedido"
        elif answer.duracao_total_alvo_sec and answer.duracao_total_alvo_sec > 0:
            result.target_total_sec, result.target_origin = answer.duracao_total_alvo_sec, "ia"
        return result

    # --- E11: detalhe ---

    def _continuity(self, usage: Usage) -> list[str]:
        return [f"{sid} — {self.material.sentences[sid].text}" for sid in usage.kept_order[-CONTINUITY_SENTENCES:]]

    def _content_budget(self, block) -> int:
        """Tokens para as seções dos clipes: disponível − pedido sem seções nem acervo − mínimo do acervo."""
        longest = max((s.text for s in self.material.sentences.values()), key=len, default="")
        probe = DetailPromptInput(
            unit_id=block.id, block_id=block.id, title=block.title, subject=block.subject, context=self.context,
            target_total_sec=None, block_target_sec=block.target_sec, used_before_sec=0.0, part=1, parts=2,
            previous_sentences=[f"S000 — {longest}"] * CONTINUITY_SENTENCES,
        )
        return self.budget.detail_available - tokens(build_detail_prompt(probe)) - LIBRARY_MIN_TOKENS

    def _units(self, block) -> list[Unit]:
        """O bloco inteiro, ou pedaços em ordem: grupos de clipes e, num clipe gigante, grupos de momentos."""
        material = self.material
        budget = self._content_budget(block)
        sizes = {c: tokens(material.section(c) + "\n") for c in block.clip_ids}
        if sum(sizes.values()) <= budget:
            return [Unit(unit_id=block.id, block=block, clip_ids=list(block.clip_ids))]
        groups: list[tuple[list[str], Optional[list[str]]]] = []
        current: list[str] = []
        used = 0
        for clip_id in block.clip_ids:
            if sizes[clip_id] > budget:
                if current:
                    groups.append((current, None))
                    current, used = [], 0
                groups += [([clip_id], chunk) for chunk in self._moment_chunks(clip_id, budget)]
                continue
            if current and used + sizes[clip_id] > budget:
                groups.append((current, None))
                current, used = [], 0
            current.append(clip_id)
            used += sizes[clip_id]
        if current:
            groups.append((current, None))
        letters = "abcdefghijklmnopqrstuvwxyz"
        return [
            Unit(
                unit_id=f"{block.id}{letters[k] if k < len(letters) else k + 1}", block=block, clip_ids=clips,
                moment_ids=moments, part=k + 1, parts=len(groups),
            )
            for k, (clips, moments) in enumerate(groups)
        ]

    def _moment_chunks(self, clip_id: str, budget: int) -> list[list[str]]:
        chunks: list[list[str]] = []
        current: list[str] = []
        for moment in self.material.clip_moments.get(clip_id, []):
            trial = current + [moment.id]
            if current and tokens(self.material.section(clip_id, trial) + "\n") > budget:
                chunks.append(current)
                trial = [moment.id]
            current = trial
        if current:
            chunks.append(current)
        return chunks

    def _unit_target(self, unit: Unit) -> Optional[float]:
        """Alvo do pedaço: o do bloco; dividido, proporcional à duração das frases (sem fala: dos momentos)."""
        target = unit.block.target_sec
        if target is None or unit.parts == 1:
            return target
        material = self.material
        block_sentences = [s.id for c in unit.block.clip_ids for s in material.clip_sentences.get(c, [])]
        whole = sentences_sec(material, block_sentences)
        part = sentences_sec(material, unit.sentence_ids(material))
        if whole <= 0:
            whole = sum(m.duration for c in unit.block.clip_ids for m in material.clip_moments.get(c, []))
            part = sum(m.duration for m in unit.moments(material))
        return round(target * part / whole, 1) if whole > 0 else target

    def _duration_problem(self, check, target: Optional[float], free_silent: int) -> Optional[str]:
        """Resultado fora de alvo ± `DURATION_TOLERANCE` (só informação; None = dentro ou sem alvo).

        Abaixo do alvo sem nada cortado e sem momento sem fala livre, não há o que ajustar: não pede de novo.
        """
        if target is None or target <= 0:
            return None
        low, high = target * (1 - DURATION_TOLERANCE), target * (1 + DURATION_TOLERANCE)
        if low <= check.result_sec <= high:
            return None
        if check.result_sec < low and not check.cuts and free_silent == 0:
            return None
        return duration_retry_line(check.result_sec, target, low, high)

    def _detail(self, unit: Unit, structure: StructureRun, usage: Usage) -> tuple[DetailRun, Usage]:
        material, context = self.material, self.context
        block = unit.block
        detail = DetailRun(
            unit_id=unit.unit_id, block_id=block.id, clip_ids=list(unit.clip_ids), moment_ids=unit.moment_ids,
            origin="ia", used_before_sec=round(usage.used_sec, 3),
        )
        self._progress(f"bloco {unit.unit_id} ({', '.join(unit.clip_ids)})")
        sentence_ids = [s for s in unit.sentence_ids(material) if s not in usage.sentences]
        moments = [m for m in unit.moments(material) if m.id not in usage.moments]
        silent = unit.silent_moments(material, usage)
        detail.available_sec = round(sentences_sec(material, sentence_ids), 3)
        detail.target_sec = self._unit_target(unit)
        data = DetailPromptInput(
            unit_id=unit.unit_id, block_id=block.id, title=block.title, subject=block.subject, context=context,
            target_total_sec=structure.target_total_sec, block_target_sec=block.target_sec,
            used_before_sec=usage.used_sec, part=unit.part, parts=unit.parts,
            previous_sentences=self._continuity(usage),
            sections=[material.section(c, unit.moment_ids) for c in unit.clip_ids],
            silent_lines=[material.moment_line(m) for m in silent], available_sec=detail.available_sec,
            unit_target_sec=detail.target_sec, clip_ids=list(unit.clip_ids),
        )
        library_budget = self.budget.detail_available - tokens(build_detail_prompt(data))
        if library_budget < 0:
            detail.warnings.append(
                f"o pedido passa do contexto em ~{-library_budget} tokens mesmo dividido (acervo vazio)"
            )
        candidates = [
            m for clip_id in structure.library_clip_ids for m in material.clip_moments.get(clip_id, [])
            if m.id not in usage.moments and not usage.conflict(m.clip_id, m.start_sec, m.end_sec)
        ]
        offered, spent = [], 0
        for moment in library_ranking(material, unit.clip_ids, candidates):
            cost = tokens(material.moment_line(moment) + "\n")
            if spent + cost > library_budget:
                break
            offered.append(moment)
            spent += cost
        if len(offered) < len(candidates):
            detail.warnings.append(f"acervo: {len(offered)} de {len(candidates)} momentos couberam no pedido")
        detail.library_offered = [m.id for m in offered]
        data.library_lines = [material.moment_line(m) for m in offered]
        prompt = build_detail_prompt(data)
        detail.estimated_tokens = tokens(prompt)
        if not sentence_ids and not moments:
            detail.origin = "fallback"
            detail.warnings.append("nada livre neste pedaço (frases e momentos já usados)")
            self.calls_done += 1
            return detail, usage

        key = self.key("detalhe", DETAIL_PROMPT_VERSION, prompt)
        check = None
        accepted: Optional[BlockAnswer] = None
        duration_problem: Optional[str] = None
        cached = self.cache.load("detalhe", key, BlockAnswer)
        if cached is not None:
            trial = validate_detail(cached, unit, material, detail.library_offered, usage)
            if trial.unusable is None:
                check, accepted, detail.origin = trial, cached, "cache"
                duration_problem = self._duration_problem(trial, detail.target_sec, len(silent))
                detail.stats.cached += 1
        errors: list[str] = []
        for _ in range(0 if check is not None else MAX_ATTEMPTS):
            request = DetailRequest(
                unit_id=unit.unit_id, prompt=build_detail_prompt(data, errors), material=material, context=context,
                clip_ids=list(unit.clip_ids), sentence_ids=sentence_ids, moments=moments, library=offered,
                target_sec=structure.target_total_sec, block_target_sec=detail.target_sec, silent=silent,
            )
            raw, error = self._ask(self.ai.detail, request, detail.stats)
            if raw is not None:
                trial = validate_detail(raw, unit, material, detail.library_offered, usage)
                if trial.unusable is None:  # aproveitável: vale se a nova tentativa não resolver
                    check, accepted = trial, raw
                    duration_problem = self._duration_problem(trial, detail.target_sec, len(silent))
                problems = [p for p in (trial.refused, duration_problem if trial.unusable is None else None) if p]
                if not problems:
                    break
                error = "; ".join(problems) + (": " + "; ".join(trial.errors[:12]) if trial.errors else "")
            errors.append(error)
            detail.attempts.append(error)
            if self.gave_up:
                break
        if accepted is not None:
            detail.notes = accepted.notas
            if detail.origin != "cache":
                self.cache.save("detalhe", key, accepted)
            if detail.attempts and check.discarded:
                detail.warnings.append(
                    f"aceitos os itens válidos da resposta; {len(check.discarded)} item(ns) inválido(s) descartado(s)"
                )
            if duration_problem:
                detail.duration_off = True
                detail.warnings.append(f"fora do alvo depois da nova tentativa, aceito assim ({duration_problem})")
        if check is None:
            check = validate_detail(neutral_detail(unit, material, usage), unit, material, [], usage)
            detail.origin = "fallback"
            detail.notes = "fallback neutro"
            message = (
                f"BLOCO {unit.unit_id} PELO FALLBACK NEUTRO (a IA falhou de vez): todas as frases dos clipes em ordem, "
                f"sem cobertura nem respiros. Motivo: {detail.attempts[-1] if detail.attempts else 'sem resposta'}"
            )
            detail.warnings.append(message)
            self._notice(message)
        self.calls_done += 1
        detail.keep, detail.coverages, detail.breaths = check.keep, check.coverages, check.breaths
        detail.cuts, detail.clip_order, detail.result_sec = check.cuts, check.clip_order, check.result_sec
        detail.discarded = check.discarded
        if detail.origin != "fallback":
            detail.normalized_ids = check.normalized
        detail.warnings += check.warnings
        for item in check.discarded:
            detail.warnings.append(f"{item.kind} descartado ({item.item}): {item.reason}")
        return detail, check.usage


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
