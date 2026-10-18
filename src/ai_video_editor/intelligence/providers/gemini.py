"""Adaptador para a API do Google Gemini utilizando o SDK oficial google-genai."""

import json
import os
from pathlib import Path
from typing import Optional
from PIL import Image

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata
from ai_video_editor.intelligence.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AIResponseValidationError,
    AITimeoutError,
    AIUnavailableError,
)
from ai_video_editor.intelligence.provider import MultimodalAIProvider
from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    VisualAnalysisResponse,
)


def resolve_gemini_api_key() -> str:
    """Obtém a chave da API do Gemini prioritariamente de variável de ambiente.

    Suporta opcionalmente leitura de arquivo .env local sem expor chaves em logs.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # Suporte opcional a arquivo .env local (ignorado pelo Git)
    env_file = Path(".env")
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass

    raise AIAuthenticationError(
        "gemini",
        details="Variável de ambiente GEMINI_API_KEY não encontrada.",
    )


def _as_int(value) -> Optional[int]:
    """Converte contagem de tokens do SDK para int, ignorando valores ausentes ou inesperados."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class GeminiMultimodalProvider(MultimodalAIProvider):
    """Adaptador de IA multimodal conectando ao Google Gemini via SDK oficial."""

    def __init__(self, api_key: Optional[str] = None):
        # A chave é resolvida sob demanda para permitir mockar o ambiente facilmente
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        """Instancia o cliente do SDK google-genai de forma tardia (lazy)."""
        from google import genai

        key = self._api_key or resolve_gemini_api_key()
        return genai.Client(api_key=key)

    def analyze_frames(
        self,
        metadata: VideoMetadata,
        frames: list[ExtractedFrame],
        config: AIAnalysisConfig,
    ) -> VisualAnalysisResponse:
        """Envia imagens e prompt estruturado ao Gemini, solicitando resposta em JSON Pydantic."""
        from google import genai
        from google.genai import types

        self.last_input_tokens = None
        self.last_output_tokens = None
        client = self._get_client()

        # Constrói instruções anti-alucinação estritas
        frame_manifest = "\n".join(
            f"- Frame #{f.frame_index}: timestamp {f.target_timestamp_sec:.2f}s (arquivo: {f.image_path.name})"
            for f in frames
        )

        prompt_text = (
            f"Você é um assistente especialista em análise técnica e visual de vídeo.\n"
            f"Analise a sequência de {len(frames)} frames amostrados do arquivo '{metadata.source_path.name}'.\n"
            f"Duração total do vídeo original: {metadata.duration_sec:.2f} segundos.\n\n"
            f"MANIFESTO DOS FRAMES ENVIADOS (use estritamente estes índices e timestamps):\n"
            f"{frame_manifest}\n\n"
            f"DIRETRIZES OBRIGATÓRIAS:\n"
            f"1. Para CADA frame da lista, preencha 'chronological_timeline' com 'frame_index' e 'target_timestamp_sec' exatos.\n"
            f"2. 'observable_content': descreva apenas o que é comprovadamente visível na imagem, sem suposições.\n"
            f"3. 'inferences': separe em hipóteses qualquer dedução de intenção, contexto narrativo ou acontecimentos não visíveis.\n"
            f"4. 'scene_transitions': se notar mudanças bruscas de conteúdo entre dois frames vizinhos, aponte a hipótese. "
            f"Como a amostragem é esparsa, 'is_confirmed_exact' DEVE ser false.\n"
            f"5. 'limitations_and_blindspots': liste os pontos cegos decorrentes dos intervalos não amostrados entre os frames.\n"
        )

        # Prepara as partes do conteúdo multimodal (texto + imagens PIL)
        contents = [prompt_text]
        for f in frames:
            img = Image.open(f.image_path)
            contents.append(f"\n[Imagem do Frame {f.frame_index} em {f.target_timestamp_sec:.2f}s]:")
            contents.append(img)

        try:
            response = client.models.generate_content(
                model=config.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VisualAnalysisResponse,
                    temperature=config.temperature,
                ),
            )
        except Exception as e:
            err_str = str(e).lower()
            # Mapeamento seguro de erros sem expor credenciais
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "permission" in err_str or "api_key" in err_str:
                raise AIAuthenticationError("gemini", details="Credencial inválida ou sem permissão de acesso.")
            elif "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                raise AIRateLimitError("gemini", details="Cota de requisições excedida.")
            elif "timeout" in err_str or "timed out" in err_str:
                raise AITimeoutError("gemini", timeout_sec=config.timeout_sec)
            elif "500" in err_str or "503" in err_str or "unavailable" in err_str:
                raise AIUnavailableError("gemini", details="Serviço do Google Gemini temporariamente indisponível.")
            else:
                raise AIProviderError(f"Erro na chamada do Gemini: {e}", provider="gemini")

        usage = getattr(response, "usage_metadata", None)
        self.last_input_tokens = _as_int(getattr(usage, "prompt_token_count", None))
        self.last_output_tokens = _as_int(getattr(usage, "candidates_token_count", None))

        # Converte a resposta estruturada para o modelo Pydantic
        try:
            raw_text = response.text
            data = json.loads(raw_text)
            validated = VisualAnalysisResponse.model_validate(data)
            # Assegura que provider_used e model_used correspondam aos parâmetros executados
            validated.provider_used = "gemini"
            validated.model_used = config.model_id
            return validated
        except Exception as e:
            raise AIResponseValidationError("gemini", details=f"Falha ao validar JSON do modelo: {e}")
