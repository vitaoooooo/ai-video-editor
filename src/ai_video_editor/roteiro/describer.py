"""Descrição de cada foto por uma IA de visão (padrão: Qwen2.5-VL-3B local, via Ollama).

Uma chamada por foto: a imagem, a fala daquele momento (palavras do Whisper até a próxima foto, ou
"sem fala") e a descrição da foto mantida anterior do mesmo clipe (para manter os nomes coerentes).
A resposta (`PhotoDescription`) é validada; recusada, a foto é pedida de novo com o erro avisado; se
persistir, fica sem descrição (aviso) e é pedida de novo na próxima execução (só o que deu certo vai
para o cache).

Cache por foto em `<projects>/_cache/roteiro_fotos/<fingerprint>/<hash da imagem>.json`, fingerprint =
versão do prompt + cadeia de modelos pedida + raciocínio. A fala e a descrição anterior não entram na
chave: a mesma imagem com o mesmo modelo não é descrita de novo.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIResponseValidationError,
)
from ai_video_editor.llm.chain import ModelChain
from ai_video_editor.llm.schema import ImageInput
from ai_video_editor.llm.stage import ChainBackedStage
from ai_video_editor.roteiro.schema import PhotoDescription

# Sobe quando o prompt ou o schema da resposta mudam (invalida as descrições em cache).
VISION_PROMPT_VERSION = 1
# Tentativas por foto (a 1ª + correções com o erro avisado).
MAX_PHOTO_ATTEMPTS = 3
# Fala no prompt: no máximo isso de palavras (o resto vira "…").
MAX_SPEECH_WORDS = 60
VISION_CACHE_DIR = Path("_cache") / "roteiro_fotos"

# --- estimativa (`--so-estimar`; ordem de grandeza, a calibrar com a medição real) ---
# Qwen2.5-VL: blocos de 28 px → ~190 tokens numa foto 512x288; texto do prompt ~350; resposta ~120.
IMAGE_TOKENS_ESTIMATE = 190
PROMPT_TOKENS_ESTIMATE = 350
OUTPUT_TOKENS_ESTIMATE = 120
# Segundos por foto do Qwen2.5-VL-3B numa GPU de 6 GB (RTX 2060), sem medição ainda.
SECONDS_PER_PHOTO_ESTIMATE = 3.0


@dataclass
class PhotoRequest:
    """Uma foto a descrever: o que vai no prompt e a imagem."""

    photo_id: str
    clip_id: str
    role: str
    image: Path
    speech: str  # vazio = sem fala
    previous: Optional[PhotoDescription] = None


@dataclass
class PhotoOutcome:
    description: Optional[PhotoDescription] = None
    error: str = ""
    api_calls: int = 0
    corrections: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: Optional[str] = None
    warnings: list[str] = field(default_factory=list)  # avisos do provedor (uma vez por processo)


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:20]


def short_speech(words: list[str]) -> str:
    text = " ".join(words)
    if len(words) > MAX_SPEECH_WORDS:
        text = " ".join(words[:MAX_SPEECH_WORDS]) + " …"
    return text


def describe_line(description: PhotoDescription) -> str:
    """Resumo de uma descrição numa linha (prompt da foto seguinte)."""
    parts = [description.what]
    if description.place:
        parts.append(f"lugar: {description.place}")
    parts.append(f"tags: {', '.join(description.tags)}")
    return " | ".join(parts)


def build_vision_prompt(request: PhotoRequest, previous_error: Optional[str] = None) -> str:
    """Prompt curto e direto (o modelo padrão é pequeno)."""
    speech = f'"{request.speech}"' if request.speech else "sem fala"
    previous = (
        describe_line(request.previous) if request.previous is not None else "nenhuma (é a primeira foto do clipe)"
    )
    lines = [
        "Você descreve UMA foto tirada de um vídeo, para um editor de vídeo. Responda só com o JSON pedido, "
        "em português.",
        f"Foto {request.photo_id} (clipe {request.clip_id}, papel {request.role}; o papel é só contexto).",
        f"Fala neste momento do vídeo: {speech}",
        f"Foto anterior do mesmo clipe: {previous}",
        "Descreva o que ESTA foto mostra; use a anterior só para manter os mesmos nomes (mesmo lugar, mesma "
        "pessoa) quando for a mesma cena. A fala ajuda a entender o contexto, mas descreva a imagem.",
        "",
        "Campos:",
        "- what: o que aparece, em uma frase curta.",
        "- place: o lugar (ex.: praia, quarto de hotel, trilha na mata, rua); \"\" se não der para saber.",
        "- people: quantas pessoas aparecem (0 se nenhuma).",
        "- talking_to_camera: true se alguém olha e fala para a câmera.",
        "- action: o que está acontecendo, em poucas palavras.",
        "- tags: de 1 a 6 palavras curtas e minúsculas (coisas, lugar, ação).",
        "- issue: um de nenhum, borrada, tremida, obstruída, escura, clara, tela uniforme.",
    ]
    if previous_error:
        lines += ["", f"Sua resposta anterior foi recusada: {previous_error}", "Corrija e responda de novo."]
    return "\n".join(lines)


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _add(total: Optional[int], value: Optional[int]) -> Optional[int]:
    return total if value is None else (total or 0) + value


class PhotoDescriber(ChainBackedStage):
    """Pede `PhotoDescription` à cadeia de modelos, foto a foto, com validação e correção.

    `describe` pode ser chamado por várias threads (um clipe por thread).
    """

    def __init__(self, chain: ModelChain, on_retry: Optional[Callable[[str], None]] = None):
        super().__init__(chain)
        self.on_retry = on_retry

    @property
    def fingerprint(self) -> str:
        payload = json.dumps([VISION_PROMPT_VERSION, self.model_chain, self.chain.thinking])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def describe(self, request: PhotoRequest) -> PhotoOutcome:
        """Descreve uma foto. Erro da API (exceto autenticação) deixa a foto sem descrição.

        Raises:
            AIAuthenticationError: sem chave ou chave inválida (não adianta seguir).
        """
        outcome = PhotoOutcome()
        image = ImageInput(path=request.image, label=request.photo_id)
        error: Optional[str] = None
        for attempt in range(1, MAX_PHOTO_ATTEMPTS + 1):
            outcome.corrections = attempt - 1
            try:
                description = self._generate(build_vision_prompt(request, error), PhotoDescription, [image])
            except AIAuthenticationError:
                raise
            except (AIResponseValidationError, ValueError) as e:
                # ValueError: a validação do Pydantic fora do SDK (ex: provedor simulado).
                self._account(outcome)
                error = f"JSON inválido: {getattr(e, 'details', '') or _short(str(e))}"
            except AIProviderError as e:
                self._account(outcome)
                outcome.error = f"erro da IA: {_short(str(e))}"
                return outcome
            else:
                self._account(outcome)
                outcome.description = description
                return outcome
            if attempt < MAX_PHOTO_ATTEMPTS and self.on_retry is not None:
                self.on_retry(f"{request.photo_id}: resposta recusada; nova tentativa {attempt + 1}/{MAX_PHOTO_ATTEMPTS}")
        outcome.error = f"resposta inválida após {MAX_PHOTO_ATTEMPTS} tentativas: {_short(error or '')}"
        return outcome

    def _account(self, outcome: PhotoOutcome) -> None:
        outcome.api_calls += self.last_api_attempts
        outcome.input_tokens = _add(outcome.input_tokens, self.last_input_tokens)
        outcome.output_tokens = _add(outcome.output_tokens, self.last_output_tokens)
        metrics = self.chain.last_metrics
        if metrics is not None:
            outcome.model = f"{metrics.provider}:{metrics.model_id}"
            outcome.warnings += metrics.warnings


# --- cache por foto ---


def cache_file(projects_root: Path, fingerprint: str, digest: str) -> Path:
    return Path(projects_root).resolve() / VISION_CACHE_DIR / fingerprint / f"{digest}.json"


def load_cached(path: Path) -> Optional[PhotoDescription]:
    if not path.is_file():
        return None
    try:
        return PhotoDescription.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def save_cached(path: Path, description: PhotoDescription) -> None:
    """Temporário + troca: um Ctrl+C no meio da escrita não corrompe o cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temp.write_text(description.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp, path)


# --- provedor simulado ---

_COLORS = {
    "vermelho": (220, 30, 30), "verde": (30, 200, 30), "azul": (30, 30, 220), "branco": (240, 240, 240),
    "preto": (15, 15, 15), "amarelo": (230, 230, 30), "cinza": (128, 128, 128),
}


def dominant_color(path: Path) -> str:
    """Cor média da imagem, pelo nome mais próximo (o mock descreve fotos sintéticas pela cor)."""
    with Image.open(path) as img:
        r, g, b = img.convert("RGB").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    return min(_COLORS, key=lambda name: sum((c - v) ** 2 for c, v in zip(_COLORS[name], (r, g, b))))


def mock_vision_responder(prompt: str, images: list[ImageInput], schema) -> PhotoDescription:
    """Resposta simulada e determinística (para `--visao-provider mock` e testes): descreve pela cor
    média da imagem; fotos da mesma cor ficam com o mesmo lugar e as mesmas tags."""
    color = dominant_color(images[0].path) if images else "cinza"
    speaking = "sem fala" not in prompt.split("Fala neste momento do vídeo:", 1)[-1].split("\n", 1)[0]
    return PhotoDescription(
        what=f"Imagem simulada predominantemente {color}.",
        place=f"cenário {color}",
        people=1 if speaking else 0,
        talking_to_camera=speaking,
        action="pessoa falando" if speaking else "cena parada",
        tags=[color, "simulada"],
        issue="tela uniforme" if color in ("preto", "branco") else "nenhum",
    )
