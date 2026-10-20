"""Adaptador para a API do Google Gemini utilizando o SDK oficial google-genai."""

import json
from typing import Optional
from PIL import Image

from ai_video_editor.analyzer.schema import ExtractedFrame, VideoMetadata
from ai_video_editor.intelligence.exceptions import AIResponseValidationError
from ai_video_editor.intelligence.provider import MultimodalAIProvider
from ai_video_editor.intelligence.retry import RetryPolicy, call_with_retry
from ai_video_editor.intelligence.schema import (
    AIAnalysisConfig,
    VisualAnalysisResponse,
)
# A lógica comum do Gemini (chave, timeout, mapeamento de erros) mora na camada `llm`.
from ai_video_editor.llm.client import as_int as _as_int
from ai_video_editor.llm.providers.gemini import (  # noqa: F401 — reexportados
    build_http_options,
    map_gemini_error,
    resolve_gemini_api_key,
)


class GeminiMultimodalProvider(MultimodalAIProvider):
    """Adaptador de IA multimodal conectando ao Google Gemini via SDK oficial."""

    def __init__(self, api_key: Optional[str] = None, retry: Optional[RetryPolicy] = None):
        # A chave é resolvida sob demanda para permitir mockar o ambiente facilmente
        self._api_key = api_key
        self.retry = retry if retry is not None else RetryPolicy()

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
        self.last_api_attempts = 1
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

        self.retry.reset()
        try:
            response = call_with_retry(
                lambda: client.models.generate_content(
                    model=config.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VisualAnalysisResponse,
                        temperature=config.temperature,
                        http_options=build_http_options(config.timeout_sec),
                    ),
                ),
                self.retry,
                lambda e: map_gemini_error(e, config.timeout_sec),
            )
        finally:
            self.last_api_attempts = max(1, self.retry.attempts_made)

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
