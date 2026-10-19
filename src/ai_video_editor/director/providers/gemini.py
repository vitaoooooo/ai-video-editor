"""Diretor de IA com o Google Gemini (só texto, saída JSON estruturada)."""

import json
from typing import Optional

from ai_video_editor.director.provider import EditDirector
from ai_video_editor.director.schema import (
    DirectorContext,
    DirectorDecision,
    DirectorThinking,
    Sentence,
    UserBrief,
)
from ai_video_editor.director.service import allowed_removal_ratio, speech_duration
from ai_video_editor.intelligence.exceptions import AIProviderError, AIResponseValidationError
from ai_video_editor.intelligence.providers.gemini import (
    _as_int,
    build_http_options,
    map_gemini_error,
    resolve_gemini_api_key,
)

DEFAULT_DIRECTOR_MODEL = "gemini-3.8-flash"

# Família 3.x: `thinking_level`. Família 2.5 (legado): `thinking_budget` em tokens (-1 = dinâmico).
_THINKING_LEVELS = {"off": "MINIMAL", "low": "LOW", "high": "HIGH"}
_THINKING_BUDGETS = {"off": 0, "low": 1024, "high": -1}


def _thinking_setting(model_id: str, thinking: DirectorThinking) -> tuple[str, int | str]:
    if "gemini-2.5" in model_id:
        return "thinking_budget", _THINKING_BUDGETS[thinking]
    return "thinking_level", _THINKING_LEVELS[thinking]


def thinking_label(model_id: str, thinking: DirectorThinking) -> str:
    """Descrição da configuração de raciocínio, para o relatório."""
    name, value = _thinking_setting(model_id, thinking)
    return f"{thinking} ({name}={value})"


def build_thinking_config(model_id: str, thinking: DirectorThinking):
    """ThinkingConfig do SDK para o nível pedido."""
    from google.genai import types

    name, value = _thinking_setting(model_id, thinking)
    return types.ThinkingConfig(**{name: value})


_THINKING_ORDER: tuple[DirectorThinking, ...] = ("off", "low", "high")


def _fallback_ladder(model_id: str, thinking: DirectorThinking) -> list[DirectorThinking]:
    """Níveis a tentar, do pedido para cima (off → low → high). Só a família 3.x tem níveis."""
    if "gemini-2.5" in model_id:
        return [thinking]
    return list(_THINKING_ORDER[_THINKING_ORDER.index(thinking):])


def is_thinking_level_unsupported(error: Exception) -> bool:
    """True se o erro do SDK é a recusa 400 do nível de raciocínio pelo modelo."""
    text = str(error).lower()
    return "thinking level" in text and "not supported" in text and (
        "400" in text or "invalid_argument" in text
    )


_INTENSITY_GUIDE = {
    "light": (
        "LEVE: remova apenas problemas evidentes — frases recomeçadas, repetições quase "
        "literais e enrolação sem conteúdo. Na maioria dos vídeos isso é pouco ou nada."
    ),
    "medium": (
        "MÉDIA: além dos problemas evidentes, remova repetições de ideias já ditas e "
        "digressões curtas que não acrescentam nada ao assunto."
    ),
    "aggressive": (
        "AGRESSIVA: deixe o vídeo enxuto; remova também desvios do assunto e explicações "
        "redundantes, mas mantenha toda informação necessária para entender o conteúdo."
    ),
}


def build_director_prompt(sentences: list[Sentence], brief: UserBrief, context: DirectorContext) -> str:
    """Monta o prompt do Diretor: regras, briefing do usuário e frases numeradas."""
    total = speech_duration(sentences)
    limit = allowed_removal_ratio(sentences, brief)
    lines = [
        "Você é um editor de vídeo experiente revisando a transcrição de um vídeo falado "
        "(aula, tutorial ou pessoa falando para a câmera).",
        f"Vídeo: '{context.video_name}', duração {context.duration_sec:.1f}s, "
        f"idioma {context.language or 'desconhecido'}. Tempo de fala nas frases: {total:.1f}s.",
        "",
        "TAREFA: escolha quais frases devem ser REMOVIDAS do vídeo. As pausas silenciosas já "
        "são cortadas automaticamente por outra etapa; não se preocupe com elas.",
        "",
        "REGRAS OBRIGATÓRIAS:",
        "1. Seja conservador. Na dúvida, MANTENHA a frase.",
        "2. Nunca remova uma frase que carrega informação nova (um passo, um conceito, um dado, "
        "um exemplo, uma conclusão), mesmo que esteja mal falada.",
        "3. Em repetições e frases recomeçadas, remova a versão pior ou incompleta e mantenha a "
        "melhor (normalmente a última, completa).",
        "4. Referencie as frases SOMENTE pelo ID entre colchetes. Não invente IDs e não "
        "escreva tempos ou timestamps.",
        "5. Cada ID pode aparecer no máximo uma vez.",
        f"6. Não remova mais que {limit:.0%} do tempo de fala.",
        "7. Escreva 'reason' e 'summary' em português, de forma curta e específica.",
        "",
        "CATEGORIAS: repetition (repete algo já dito), false_start (frase começada e "
        "abandonada ou refeita logo depois), filler (enrolação sem conteúdo), off_topic "
        "(fora do assunto do vídeo), other (outro motivo; explique).",
        "",
        "BRIEFING DO USUÁRIO:",
        f"- Intensidade: {_INTENSITY_GUIDE[brief.intensity]}",
    ]
    if brief.target_duration_sec is not None:
        lines.append(
            f"- Duração final desejada: cerca de {brief.target_duration_sec:.0f}s. Corte o necessário "
            "para se aproximar disso, priorizando o que tem menos informação."
        )
    if not brief.remove_fillers:
        lines.append("- NÃO remova frases por serem enrolação: a categoria 'filler' está proibida.")
    if brief.instructions:
        lines.append(f"- Pedido do usuário: {brief.instructions.strip()}")
    if context.previous_error:
        lines += [
            "",
            "ATENÇÃO: sua resposta anterior foi rejeitada por este motivo:",
            f"{context.previous_error}",
            "Corrija o problema nesta nova resposta.",
        ]
    lines += ["", "FRASES (ID, duração e texto):"]
    lines += [f"[{s.id}] ({s.duration:.1f}s) {s.text}" for s in sentences]
    lines += ["", "Responda em JSON com 'removals' (pode ser vazia) e 'summary'."]
    return "\n".join(lines)


class GeminiDirector(EditDirector):
    """Diretor usando o Gemini com `response_schema` = `DirectorDecision`."""

    def __init__(
        self,
        model_id: str = DEFAULT_DIRECTOR_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout_sec: float = 120.0,
        thinking: DirectorThinking = "low",
        client=None,
    ):
        self._model_id = model_id
        self._api_key = api_key
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.thinking = thinking
        self._client = client
        # `thinking_label` é o nível PEDIDO (entra no fingerprint do cache);
        # `effective_thinking_label` só muda quando a API recusa o nível e subimos de nível.
        self.thinking_label = thinking_label(model_id, thinking)
        self.effective_thinking_label = self.thinking_label

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_client(self):
        """Instancia o cliente do SDK google-genai sob demanda."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key or resolve_gemini_api_key())
        return self._client

    def decide(
        self, sentences: list[Sentence], brief: UserBrief, context: DirectorContext
    ) -> DirectorDecision:
        from google.genai import types

        self.last_input_tokens = None
        self.last_output_tokens = None
        self.effective_thinking_label = self.thinking_label
        client = self._get_client()
        prompt = build_director_prompt(sentences, brief, context)

        ladder = _fallback_ladder(self._model_id, self.thinking)
        refused: list[str] = []
        response = None
        for level in ladder:
            name, value = _thinking_setting(self._model_id, level)
            try:
                response = client.models.generate_content(
                    model=self._model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DirectorDecision,
                        temperature=self.temperature,
                        thinking_config=build_thinking_config(self._model_id, level),
                        http_options=build_http_options(self.timeout_sec),
                    ),
                )
            except Exception as e:
                if not is_thinking_level_unsupported(e):
                    raise map_gemini_error(e, self.timeout_sec) from e
                refused.append(str(value))
                continue
            if refused:
                self.effective_thinking_label = (
                    f"{self.thinking} → {value} ({', '.join(refused)} não suportado)"
                )
            break
        else:
            raise AIProviderError(
                f"O modelo '{self._model_id}' recusou todos os níveis de raciocínio "
                f"tentados ({', '.join(refused)}).",
                provider="gemini",
            )

        usage = getattr(response, "usage_metadata", None)
        self.last_input_tokens = _as_int(getattr(usage, "prompt_token_count", None))
        self.last_output_tokens = _as_int(getattr(usage, "candidates_token_count", None))

        try:
            return DirectorDecision.model_validate(json.loads(response.text))
        except Exception as e:
            raise AIResponseValidationError("gemini", details=f"JSON inválido do modelo: {e}") from e
