import os
import re
import time
from collections import deque
from datetime import date

import requests
import tiktoken

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


class TokenBudgetExceeded(RuntimeError):
    pass


class LLMClient:
    """Client for any OpenAI-compatible chat completions endpoint (Ollama, Groq, OpenAI, vLLM, ...).

    Defaults to a local Ollama server. Use LLMClient.groq() for the hosted setup.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5:4b",
        endpoint: str = "http://localhost:11434/v1/chat/completions",
        api_key: str | None = None,
        max_tokens_per_day: int | None = None,
        max_queries_per_minute: int | None = None,
        temperature: float = 0.0,
        timeout: float = 300.0,
        json_mode: bool = True,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode

        # Local servers usually need no key; hosted APIs do.
        key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"

        self._session = requests.Session()
        self._request_times: deque[float] = deque()
        self._tokens_used_today = 0
        self._budget_day = date.today()

        try:
            self._encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Approximation for non-OpenAI models; fine for budget estimates.
            self._encoder = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def groq(
        cls,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        **kwargs,
    ) -> "LLMClient":
        """Hosted Groq setup with its free-tier limits."""
        kwargs.setdefault("max_tokens_per_day", 500_000)
        kwargs.setdefault("max_queries_per_minute", 30)
        kwargs.setdefault("timeout", 60.0)
        return cls(
            model_name=model_name,
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key=api_key,
            **kwargs,
        )

    def query(self, user_input: str, system_message: str) -> str:
        """Queries the LLM server for user_input."""
        estimated = len(self._encoder.encode(user_input)) + len(
            self._encoder.encode(system_message)
        )
        self._check_token_budget(estimated)
        self._wait_for_rate_limit()

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = self._session.post(
            self.endpoint, headers=self._headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        result: dict = resp.json()

        usage = result.get("usage") or {}
        self._tokens_used_today += usage.get("total_tokens", estimated)

        choices = result.get("choices") or []
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content") or ""
        # Reasoning models may prepend a <think> block; it can contain braces
        # that confuse the JSON extraction downstream.
        return _THINK_BLOCK.sub("", content).strip()

    def _check_token_budget(self, estimated: int) -> None:
        if self.max_tokens_per_day is None:
            return
        today = date.today()
        if today != self._budget_day:
            self._budget_day = today
            self._tokens_used_today = 0
        if self._tokens_used_today + estimated > self.max_tokens_per_day:
            raise TokenBudgetExceeded(
                f"Daily token budget of {self.max_tokens_per_day} would be exceeded "
                f"({self._tokens_used_today} used, ~{estimated} needed)."
            )

    def _wait_for_rate_limit(self) -> None:
        if self.max_queries_per_minute is None:
            return
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] >= 60:
            self._request_times.popleft()
        if len(self._request_times) >= self.max_queries_per_minute:
            time.sleep(60 - (now - self._request_times[0]))
        self._request_times.append(time.monotonic())
