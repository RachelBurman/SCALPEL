"""
LLM Client for SCALPEL.

Wrapper for Ollama API with streaming and structured output support.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import ollama
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
        """Approximate tokens per character ratio."""
        if not self.content:
            return 0
        return self.completion_tokens / len(self.content)


class LLMClient:
    """
    Client for Ollama LLM interactions.
    
    Provides both streaming and non-streaming responses with
    token counting and rich console output.
    """
    
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        context_length: int | None = None,
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Ollama model name (default from settings)
            temperature: Generation temperature (default from settings)
            context_length: Max context window (default from settings)
        """
        self.model = model or settings.ollama_model
        self.temperature = temperature if temperature is not None else settings.model_temperature
        self.context_length = context_length or settings.model_context_length
        self._client = ollama.Client(host=settings.ollama_host)
    
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        stream: bool = False,
    ) -> LLMResponse | Generator[str, None, LLMResponse]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            LLMResponse or generator yielding chunks then final response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": self.temperature,
            "num_ctx": self.context_length,
        }
        
        if stream:
            return self._stream_generate(messages, options)
        else:
            return self._generate(messages, options)
    
    def _generate(self, messages: list[dict], options: dict) -> LLMResponse:
        """Non-streaming generation."""
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        
        return LLMResponse(
            content=response["message"]["content"],
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        )
    
    def _stream_generate(
        self,
        messages: list[dict],
        options: dict,
    ) -> Generator[str, None, LLMResponse]:
        """Streaming generation yielding chunks."""
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        stream = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
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
    
    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> LLMResponse | Generator[str, None, LLMResponse]:
        """
        Multi-turn chat with message history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            stream: Whether to stream the response
        """
        options = {
            "temperature": self.temperature,
            "num_ctx": self.context_length,
        }
        
        if stream:
            return self._stream_generate(messages, options)
        else:
            return self._generate(messages, options)
    
    def generate_with_live_display(
        self,
        prompt: str,
        system: str | None = None,
        format_as: Literal["markdown", "plain"] = "markdown",
    ) -> LLMResponse:
        """
        Generate response with live streaming display in terminal.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            format_as: Display format ("markdown" or "plain")
            
        Returns:
            Complete LLMResponse after streaming finishes
        """
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
        """Check if the Ollama server and model are available."""
        try:
            response = self._client.list()
            if hasattr(response, "models"):
                model_names = [m.model.split(":")[0] for m in response.models]
            else:
                model_names = [m["name"].split(":")[0] for m in response.get("models", [])]
            return self.model.split(":")[0] in model_names
        except Exception:
            return False
    
    def get_model_info(self) -> dict | None:
        """Get information about the current model."""
        try:
            return self._client.show(self.model)
        except Exception:
            return None


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
) -> LLMResponse | Generator[str, None, LLMResponse]:
    """Generate using the default client."""
    return get_client().generate(prompt, system=system, stream=stream)


if __name__ == "__main__":
    console.print("[bold]🔪 SCALPEL LLM Client Test[/bold]\n")
    
    client = LLMClient()
    
    if not client.is_available():
        console.print(f"[red]✗ Model '{client.model}' not available![/red]")
        console.print("[dim]Make sure Ollama is running and the model is pulled[/dim]")
    else:
        console.print(f"[green]✓[/green] Model: [cyan]{client.model}[/cyan]")
        console.print(f"[green]✓[/green] Temperature: [cyan]{client.temperature}[/cyan]")
        console.print(f"[green]✓[/green] Context length: [cyan]{client.context_length:,}[/cyan]\n")
        
        console.print("[bold]Testing generation...[/bold]\n")
        
        response = client.generate_with_live_display(
            "Explain what a p-value is in one paragraph.",
            system="You are a statistics expert. Be concise and precise.",
        )
        
        console.print(f"\n[dim]Tokens: {response.prompt_tokens} prompt + "
                     f"{response.completion_tokens} completion = {response.total_tokens} total[/dim]")