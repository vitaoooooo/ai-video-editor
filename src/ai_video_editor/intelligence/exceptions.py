"""Exceções de domínio para a camada de inteligência e provedores multimodais de IA."""

from typing import Optional


class AIProviderError(Exception):
    """Exceção base para falhas de comunicação ou execução de IA multimodal."""

    def __init__(self, message: str, provider: str = "", details: str = ""):
        # Sanitiza a mensagem para garantir que nenhuma chave acidental seja exibida
        sanitized_msg = message
        super().__init__(f"[{provider.upper()}] {sanitized_msg}" if provider else sanitized_msg)
        self.provider = provider
        self.details = details


class AIAuthenticationError(AIProviderError):
    """Lançada quando a chave de API é inválida, expirou ou não foi fornecida."""

    def __init__(self, provider: str, details: str = ""):
        msg = f"Falha de autenticação com o provedor '{provider}'. Verifique se a variável de ambiente correspondente está configurada."
        super().__init__(msg, provider=provider, details=details)


class AIRateLimitError(AIProviderError):
    """Lançada quando cotas de requisição (RPM/TPM) são atingidas."""

    def __init__(self, provider: str, details: str = ""):
        msg = f"Limite de requisições ou cota excedida no provedor '{provider}'."
        super().__init__(msg, provider=provider, details=details)


class AITimeoutError(AIProviderError):
    """Lançada quando a chamada ao provedor excede o tempo limite configurado."""

    def __init__(self, provider: str, timeout_sec: float):
        msg = f"Tempo limite de {timeout_sec:.1f}s excedido durante comunicação com '{provider}'."
        super().__init__(msg, provider=provider)
        self.timeout_sec = timeout_sec


class AIUnavailableError(AIProviderError):
    """Lançada quando o serviço do provedor está temporariamente fora do ar ou com erro 5xx."""

    def __init__(self, provider: str, details: str = ""):
        msg = f"Serviço '{provider}' temporariamente indisponível."
        super().__init__(msg, provider=provider, details=details)


class AIResponseValidationError(AIProviderError):
    """Lançada quando a resposta da IA não atende ao formato ou schema Pydantic esperado."""

    def __init__(self, provider: str, details: str = ""):
        msg = f"A resposta do provedor '{provider}' não pôde ser validada de acordo com o schema."
        super().__init__(msg, provider=provider, details=details)
