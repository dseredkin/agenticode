"""LLM client abstraction with multi-provider support."""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """LLM response model."""

    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with the generated content.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI/GPT provider implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Model to use.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OpenAI API key is required")

        self._model = model
        self._client = OpenAI(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )


class GrokProvider(BaseLLMProvider):
    """xAI Grok provider implementation."""

    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-beta",
    ) -> None:
        """Initialize Grok provider.

        Args:
            api_key: xAI API key. Defaults to GROK_API_KEY env var.
            model: Model to use.
        """
        self._api_key = api_key or os.environ.get("GROK_API_KEY", "")
        if not self._api_key:
            raise ValueError("Grok API key is required")

        self._model = model
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def provider_name(self) -> str:
        return "grok"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )


class YandexProvider(BaseLLMProvider):
    """YandexGPT provider implementation."""

    BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"

    def __init__(
        self,
        api_key: str | None = None,
        folder_id: str | None = None,
        model: str = "yandexgpt-lite",
    ) -> None:
        """Initialize Yandex provider.

        Args:
            api_key: Yandex API key. Defaults to YANDEX_API_KEY env var.
            folder_id: Yandex folder ID. Defaults to YANDEX_FOLDER_ID env var.
            model: Model to use.
        """
        self._api_key = api_key or os.environ.get("YANDEX_API_KEY", "")
        self._folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID", "")

        if not self._api_key:
            raise ValueError("Yandex API key is required")
        if not self._folder_id:
            raise ValueError("Yandex folder ID is required")

        self._model = model
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def provider_name(self) -> str:
        return "yandex"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": prompt})

        model_uri = f"gpt://{self._folder_id}/{self._model}"

        response = self._client.post(
            "/completion",
            json={
                "modelUri": model_uri,
                "completionOptions": {
                    "stream": False,
                    "temperature": temperature,
                    "maxTokens": str(max_tokens),
                },
                "messages": messages,
            },
        )
        response.raise_for_status()
        data = response.json()

        result = data.get("result", {})
        alternatives = result.get("alternatives", [{}])
        content = alternatives[0].get("message", {}).get("text", "")
        usage = result.get("usage", {})

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=int(usage.get("inputTextTokens", 0)),
            completion_tokens=int(usage.get("completionTokens", 0)),
            total_tokens=int(usage.get("totalTokens", 0)),
        )


class LLMClient:
    """Factory class that selects provider based on config."""

    PROVIDERS: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "grok": GrokProvider,
        "yandex": YandexProvider,
    }

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize LLM client.

        Args:
            provider: Provider name. Defaults to LLM_PROVIDER env var.
            model: Model to use. Defaults to LLM_MODEL env var.
            max_retries: Maximum number of retries on failure.
            retry_delay: Initial delay between retries (exponential backoff).
        """
        provider_name = provider or os.environ.get("LLM_PROVIDER", "openai")
        model_name = model or os.environ.get("LLM_MODEL")

        if provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(self.PROVIDERS.keys())}"
            )

        provider_class = self.PROVIDERS[provider_name]
        kwargs: dict[str, Any] = {}
        if model_name:
            kwargs["model"] = model_name

        self._provider = provider_class(**kwargs)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._total_tokens_used = 0

    @property
    def provider(self) -> BaseLLMProvider:
        """Get the underlying provider."""
        return self._provider

    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used across all requests."""
        return self._total_tokens_used

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response with retry logic.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with the generated content.

        Raises:
            Exception: If all retries fail.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._total_tokens_used += response.total_tokens
                logger.info(
                    f"LLM response: {response.total_tokens} tokens "
                    f"(total: {self._total_tokens_used})"
                )
                return response

            except Exception as e:
                last_error = e
                delay = self._retry_delay * (2**attempt)
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self._max_retries}): "
                    f"{e}. Retrying in {delay}s..."
                )
                time.sleep(delay)

        raise last_error or Exception("All retries failed")

    def generate_code(
        self,
        prompt: str,
        context: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate code using the LLM.

        Args:
            prompt: The code generation prompt.
            context: Optional context (existing code, requirements, etc.).
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature.

        Returns:
            Generated code as a string.
        """
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"

        response = self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=4096,
        )

        return response.content
