"""Descrição das tomadas por IA, em lotes de miniaturas rotuladas com IDs.

A IA recebe as miniaturas (cada uma precedida do seu ID, ex: `P007.T03.F2`) e devolve uma
descrição por tomada. Ela nunca vê tempos: só IDs de tomadas e de miniaturas, que o código
valida e converte. Uma resposta com tomada faltando, repetida, com ID inexistente ou com
`best_frames` inválido é recusada para aquelas tomadas e o lote é refeito só com elas,
avisando o erro; se persistir, a tomada fica `sem_analise` sem derrubar o resto.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIResponseValidationError,
)
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.schema import ImageDetail, ImageInput
from ai_video_editor.llm.stage import ChainBackedStage
from ai_video_editor.visual.schema import (
    Shot,
    ShotAnalysis,
    ShotDescription,
    VisualBatchResponse,
)

# Sobe quando o prompt ou o schema da resposta mudam (invalida as análises em cache).
PROMPT_VERSION = 2

# Lote: até 8 tomadas (24 miniaturas) por chamada.
BATCH_MAX_SHOTS = 8
BATCH_MAX_IMAGES = 24
# Tentativas por lote (a 1ª + correções com o erro avisado).
MAX_BATCH_ATTEMPTS = 3
# Lotes seguidos com erro da API (não de validação) antes de desistir dos restantes.
MAX_CONSECUTIVE_FAILURES = 2
MAX_TAGS = 6

# --- estimativa (`--so-estimar`) ---
# Tokens de entrada por imagem nos modelos Gemini 3 (documentação "Media resolution"):
# padrão/HIGH = 1120, LOW = 280. Outros provedores contam diferente; é só uma ordem de grandeza.
IMAGE_TOKENS = {"low": 280, "auto": 1120, "high": 1120}
# Texto fixo do prompt por chamada e texto por tomada (ID, medidas, rótulos das imagens), aprox.
PROMPT_BASE_TOKENS = 900
SHOT_TEXT_TOKENS = 60
# Resposta por tomada (descrição, tags, problemas, notas), aprox.
OUTPUT_TOKENS_PER_SHOT = 150

RETRYABLE_HINT = "Corrija e responda de novo apenas para as tomadas listadas acima."
_MEASURES_PREFIX = "medidas: "


@dataclass
class ShotRequest:
    """Uma tomada a descrever: o que vai no prompt e as miniaturas."""

    shot: Shot
    clip_role: str
    image_paths: list  # caminhos das miniaturas, na ordem de shot.frames


@dataclass
class BatchOutcome:
    analyses: dict[str, ShotAnalysis] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)  # shot_id → motivo
    api_calls: int = 0
    images_sent: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def make_batches(requests: list[ShotRequest]) -> list[list[ShotRequest]]:
    """Lotes em ordem, com até BATCH_MAX_SHOTS tomadas e BATCH_MAX_IMAGES miniaturas."""
    batches: list[list[ShotRequest]] = []
    current: list[ShotRequest] = []
    images = 0
    for request in requests:
        count = len(request.image_paths)
        if current and (len(current) >= BATCH_MAX_SHOTS or images + count > BATCH_MAX_IMAGES):
            batches.append(current)
            current, images = [], 0
        current.append(request)
        images += count
    if current:
        batches.append(current)
    return batches


def estimate_tokens(shots: int, images: int, detail: ImageDetail) -> tuple[int, int, int]:
    """(chamadas, tokens de entrada, tokens de saída) estimados, sem novas tentativas."""
    calls = max(math.ceil(shots / BATCH_MAX_SHOTS), math.ceil(images / BATCH_MAX_IMAGES)) if shots else 0
    tokens_in = images * IMAGE_TOKENS[detail] + calls * PROMPT_BASE_TOKENS + shots * SHOT_TEXT_TOKENS
    return calls, tokens_in, shots * OUTPUT_TOKENS_PER_SHOT


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def build_visual_prompt(batch: list[ShotRequest], previous_error: Optional[str] = None) -> str:
    """Prompt do lote. Cada tomada aparece como `### <ID>` com papel, fala, medidas e miniaturas."""
    lines = [
        "Você é assistente de um editor de vídeo e descreve TOMADAS (trechos contínuos de um clipe) "
        "a partir de 3 miniaturas de cada uma: início, meio e fim. As miniaturas vêm depois deste texto, "
        "cada uma precedida do seu ID (ex: `Imagem P007.T03.F2:`).",
        "",
        "Para cada tomada, responda:",
        "- description: uma frase em português sobre o que se vê.",
        "- tags: lista curta (até 6) de palavras como paisagem, lago, praia, trilha, cidade, estrada, veículo, "
        "pessoa, rosto, animal, detalhe, céu, água, árvore, interior.",
        "- face_to_camera: true se há uma pessoa falando/olhando para a câmera (selfie, apresentador).",
        "- people_count: número aproximado de pessoas visíveis.",
        "- issues: problemas de imagem entre obstruida (parede, câmera tapada ou apontada para algo sem "
        "interesse), desfocada, tremida, escura, estourada, borrao_movimento; [\"nenhuma\"] se não houver.",
        "- usable: sim, nao ou parcial (só parte da tomada serve).",
        "- best_frames: IDs de miniaturas DESTA tomada (F1, F2, F3, em ordem) que delimitam o melhor trecho, "
        "ex: [\"P007.T03.F1\", \"P007.T03.F2\"]; vazio se nada serve.",
        "- beauty: 1 a 5, valor da imagem como cobertura (b-roll) num vídeo de viagem.",
        "- summary_for_editor: nota curta e prática para o editor.",
        "",
        "Algumas tomadas trazem uma linha `medidas:` com um alerta automático; é só dica: confie no que você vê.",
        "Responda exatamente uma entrada por tomada listada, com o `shot_id` dela; não invente IDs e não "
        "use tempos.",
        "",
        "Tomadas deste lote:",
    ]
    for request in batch:
        shot = request.shot
        speech = {True: "com fala detectada", False: "sem fala", None: "fala desconhecida"}[shot.has_speech]
        hints = shot.measures.hints if shot.measures else []
        lines.append(f"### {shot.id}")
        lines.append(
            f"papel do clipe: {request.clip_role}; duração: {_clock(shot.duration_sec)}; {speech}"
        )
        if hints:
            # Só o que foi marcado: valores de todas as medidas induziriam a IA.
            lines.append(f"{_MEASURES_PREFIX}{', '.join(hints)}")
        lines.append(f"miniaturas: {', '.join(frame.id for frame in shot.frames)}")
    if previous_error:
        lines += ["", "ATENÇÃO: a resposta anterior foi recusada:", previous_error, RETRYABLE_HINT]
    return "\n".join(lines)


def batch_images(batch: list[ShotRequest], detail: ImageDetail) -> list[ImageInput]:
    return [
        ImageInput(path=path, label=frame.id, detail=detail)
        for request in batch
        for frame, path in zip(request.shot.frames, request.image_paths)
    ]


def _normalize(item: ShotDescription) -> ShotAnalysis:
    issues = list(dict.fromkeys(item.issues))
    if len(issues) > 1 and "nenhuma" in issues:
        issues.remove("nenhuma")
    tags = list(dict.fromkeys(t.strip().lower() for t in item.tags if t.strip()))[:MAX_TAGS]
    data = item.model_dump(exclude={"shot_id"})
    data.update(issues=issues or ["nenhuma"], tags=tags,
                description=item.description.strip(), summary_for_editor=item.summary_for_editor.strip())
    return ShotAnalysis.model_validate(data)


def validate_batch(
    batch: list[ShotRequest], response: VisualBatchResponse
) -> tuple[dict[str, ShotAnalysis], list[str]]:
    """Análises aceitas por ID e a lista de problemas (vazia = lote todo aceito).

    Recusa, por tomada: faltando, respondida mais de uma vez, `best_frames` de outra tomada,
    fora de ordem ou repetido, e `best_frames` vazio numa tomada utilizável. IDs que não são do
    lote viram problema (e são ignorados).
    """
    shots = {request.shot.id: request.shot for request in batch}
    counts: dict[str, int] = {}
    for item in response.shots:
        counts[item.shot_id.strip()] = counts.get(item.shot_id.strip(), 0) + 1
    problems: list[str] = []
    unknown = [sid for sid in counts if sid not in shots]
    if unknown:
        problems.append(f"IDs de tomada inexistentes neste lote: {', '.join(unknown)}.")
    accepted: dict[str, ShotAnalysis] = {}
    for item in response.shots:
        sid = item.shot_id.strip()
        if sid not in shots or counts[sid] != 1:
            continue
        order = [frame.id for frame in shots[sid].frames]
        picked = [f.strip() for f in item.best_frames]
        foreign = [f for f in picked if f not in order]
        if foreign:
            problems.append(f"{sid}: best_frames com IDs que não são desta tomada ({', '.join(foreign)}).")
            continue
        positions = [order.index(f) for f in picked]
        if positions != sorted(set(positions)):
            problems.append(f"{sid}: best_frames fora de ordem ou repetido ({', '.join(picked)}).")
            continue
        if not picked and item.usable != "nao":
            problems.append(f"{sid}: best_frames vazio numa tomada utilizável (usable={item.usable}).")
            continue
        accepted[sid] = _normalize(item.model_copy(update={"best_frames": picked}))
    duplicated = [sid for sid, n in counts.items() if sid in shots and n > 1]
    if duplicated:
        problems.append(f"Tomadas respondidas mais de uma vez: {', '.join(duplicated)}.")
    missing = [sid for sid in shots if sid not in counts]
    if missing:
        problems.append(f"Tomadas sem resposta: {', '.join(missing)}.")
    return accepted, problems


def _add(total: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return total
    return (total or 0) + value


class VisualDescriber(ChainBackedStage):
    """Pede `VisualBatchResponse` à cadeia de modelos, lote a lote, com validação e correção."""

    def __init__(
        self,
        chain: ModelChain,
        detail: ImageDetail = "auto",
        on_retry: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(chain)
        self.detail = detail
        self.on_retry = on_retry
        self._consecutive_failures = 0

    @property
    def gave_up(self) -> bool:
        return self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def describe_batch(self, batch: list[ShotRequest]) -> BatchOutcome:
        """Descreve um lote. Erros da API (exceto autenticação) marcam o lote como sem análise.

        Raises:
            AIAuthenticationError: sem chave ou chave inválida (não adianta seguir).
        """
        outcome = BatchOutcome()
        pending = list(batch)
        error: Optional[str] = None
        if self.gave_up:
            for request in pending:
                outcome.failed[request.shot.id] = "não enviada: a IA falhou nos lotes anteriores"
            return outcome
        for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
            images = batch_images(pending, self.detail)
            try:
                response = self._generate(build_visual_prompt(pending, error), VisualBatchResponse, images)
            except AIAuthenticationError:
                raise
            except (AIResponseValidationError, ValueError) as e:
                # ValueError: a validação do Pydantic fora do SDK (ex: provedor simulado).
                self._account(outcome, len(images))
                error = f"JSON inválido: {getattr(e, 'details', '') or _short(str(e))}"
            except AIProviderError as e:
                self._account(outcome, len(images))
                self._consecutive_failures += 1
                reason = f"erro da IA: {_short(str(e))}"
                for request in pending:
                    outcome.failed[request.shot.id] = reason
                return outcome
            else:
                self._account(outcome, len(images))
                accepted, problems = validate_batch(pending, response)
                outcome.analyses.update(accepted)
                pending = [r for r in pending if r.shot.id not in accepted]
                if not pending:
                    break
                error = "\n".join(f"- {p}" for p in problems) or "- Tomadas sem resposta válida."
            if attempt < MAX_BATCH_ATTEMPTS and self.on_retry is not None:
                self.on_retry(
                    f"resposta recusada ({len(pending)} tomada(s)); nova tentativa {attempt + 1}/{MAX_BATCH_ATTEMPTS}"
                )
        self._consecutive_failures = 0
        for request in pending:
            outcome.failed[request.shot.id] = (
                f"resposta inválida após {MAX_BATCH_ATTEMPTS} tentativas: {_short(error or '')}"
            )
        return outcome

    def _account(self, outcome: BatchOutcome, images: int) -> None:
        outcome.api_calls += self.last_api_attempts
        outcome.images_sent += images
        outcome.input_tokens = _add(outcome.input_tokens, self.last_input_tokens)
        outcome.output_tokens = _add(outcome.output_tokens, self.last_output_tokens)


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- provedor simulado ---

def prompt_sections(prompt: str) -> list[tuple[str, str]]:
    """(ID da tomada, linha de medidas) de cada `### <ID>` do prompt, na ordem."""
    sections = []
    for block in prompt.split("\n### ")[1:]:
        lines = block.splitlines()
        hints = next((line[len(_MEASURES_PREFIX):] for line in lines if line.startswith(_MEASURES_PREFIX)), "")
        sections.append((lines[0].strip(), hints))
    return sections


def mock_visual_responder(prompt: str, images: list[ImageInput], schema) -> VisualBatchResponse:
    """Resposta simulada e determinística (para `--visual-provider mock` e testes).

    Lê as tomadas do prompt e as dicas das medidas: "tela uniforme" vira obstruída/inútil e
    "borrão alto" vira desfocada/parcial; o resto é utilizável.
    """
    frames_of: dict[str, list[str]] = {}
    for image in images:
        frames_of.setdefault(image.label.rsplit(".", 1)[0], []).append(image.label)
    items = []
    for shot_id, hints in prompt_sections(prompt):
        frames = frames_of.get(shot_id, [])
        if "tela uniforme" in hints:
            issues, usable, best, beauty = ["obstruida"], "nao", [], 1
        elif "borrão alto" in hints:
            issues, usable, best, beauty = ["desfocada"], "parcial", frames[:1], 2
        else:
            issues, usable, best, beauty = ["nenhuma"], "sim", [frames[0], frames[-1]] if frames else [], 3
        items.append(ShotDescription(
            shot_id=shot_id,
            description=f"Tomada simulada {shot_id}.",
            tags=["detalhe"],
            face_to_camera=False,
            people_count=0,
            issues=issues,
            usable=usable,
            best_frames=best if len(set(best)) == len(best) else best[:1],
            beauty=beauty,
            summary_for_editor="Análise simulada (mock).",
        ))
    return VisualBatchResponse(shots=items)
