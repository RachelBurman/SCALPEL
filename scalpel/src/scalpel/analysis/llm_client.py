"""
LLM Client for SCALPEL.

Supports Ollama (local) and OpenRouter (cloud) as providers.
"""

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import httpx
from rich.live import Live
from rich.markdown import Markdown

from scalpel.config import settings
from scalpel.console import console


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def tokens_per_char(self) -> float:
        if not self.content:
            return 0
        return self.completion_tokens / len(self.content)


class OllamaProvider:
    """Ollama local model provider."""

    def __init__(self, model: str, temperature: float, context_length: int):
        import ollama as _ollama
        self.model = model
        self.temperature = temperature
        self.context_length = context_length
        self._client = _ollama.Client(host=settings.ollama_host)

    def generate(self, messages: list[dict]) -> LLMResponse:
        options = {"temperature": self.temperature, "num_ctx": self.context_length}
        response = self._client.chat(model=self.model, messages=messages, options=options)
        return LLMResponse(
            content=response["message"]["content"],
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        )

    def stream(self, messages: list[dict]) -> Generator[str, None, LLMResponse]:
        options = {"temperature": self.temperature, "num_ctx": self.context_length}
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0

        stream = self._client.chat(
            model=self.model, messages=messages, options=options, stream=True
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            full_content += content
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)
            yield content

        return LLMResponse(
            content=full_content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def is_available(self) -> bool:
        try:
            import ollama as _ollama
            client = _ollama.Client(host=settings.ollama_host)
            response = client.list()
            if hasattr(response, "models"):
                model_names = [m.model.split(":")[0] for m in response.models]
            else:
                model_names = [m["name"].split(":")[0] for m in response.get("models", [])]
            return self.model.split(":")[0] in model_names
        except Exception:
            return False


class OpenRouterProvider:
    """OpenRouter cloud provider (OpenAI-compatible API)."""

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        self._base_url = settings.openrouter_base_url
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/scalpel",
            "X-Title": "SCALPEL",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict], stream: bool = False) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

    def generate(self, messages: list[dict]) -> LLMResponse:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages),
            )
            if not response.is_success:
                raise RuntimeError(
                    f"OpenRouter {response.status_code}: {response.text}"
                )
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def stream(self, messages: list[dict]) -> Generator[str, None, LLMResponse]:
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0

        with httpx.Client(timeout=120, follow_redirects=True) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, stream=True),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        yield content
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            content=full_content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def is_available(self) -> bool:
        if not settings.openrouter_api_key:
            return False
        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                r = client.get(
                    f"{self._base_url}/models",
                    headers=self._headers,
                )
                return r.status_code == 200
        except Exception:
            return False


class LLMClient:
    """
    Provider-agnostic LLM client.

    Delegates to OllamaProvider or OpenRouterProvider based on settings.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        context_length: int | None = None,
        provider: str | None = None,
    ):
        self.temperature = temperature if temperature is not None else settings.model_temperature
        self.context_length = context_length or settings.model_context_length
        chosen_provider = provider or settings.llm_provider

        if chosen_provider == "openrouter":
            self.model = model or settings.openrouter_model
            self._provider = OpenRouterProvider(self.model, self.temperature)
        else:
            self.model = model or settings.ollama_model
            self._provider = OllamaProvider(self.model, self.temperature, self.context_length)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        stream: bool = False,
    ) -> "LLMResponse | Generator[str, None, LLMResponse]":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if stream:
            return self._provider.stream(messages)
        return self._provider.generate(messages)

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> "LLMResponse | Generator[str, None, LLMResponse]":
        if stream:
            return self._provider.stream(messages)
        return self._provider.generate(messages)

    def generate_with_live_display(
        self,
        prompt: str,
        system: str | None = None,
        format_as: Literal["markdown", "plain"] = "markdown",
    ) -> LLMResponse:
        generator = self.generate(prompt, system=system, stream=True)
        full_content = ""
        final_response = None

        with Live(console=console, refresh_per_second=10) as live:
            try:
                while True:
                    chunk = next(generator)
                    full_content += chunk
                    if format_as == "markdown":
                        live.update(Markdown(full_content))
                    else:
                        live.update(full_content)
            except StopIteration as e:
                final_response = e.value

        return final_response

    def is_available(self) -> bool:
        return self._provider.is_available()


_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def generate(
    prompt: str,
    system: str | None = None,
    stream: bool = False,
) -> "LLMResponse | Generator[str, None, LLMResponse]":
    """Generate using the default client."""
    return get_client().generate(prompt, system=system, stream=stream)
