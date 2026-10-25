import os
import time
from collections import deque
from datetime import date

import requests
import tiktoken


class TokenBudgetExceeded(RuntimeError):
    pass


class LLMClient:
    """Client for any OpenAI-compatible chat completions endpoint (Groq, OpenAI, Ollama, vLLM, ...)."""

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        endpoint: str = "https://api.groq.com/openai/v1/chat/completions",
        api_key: str | None = None,
        max_tokens_per_day: int | None = 500_000,
        max_queries_per_minute: int | None = 30,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.temperature = temperature
        self.timeout = timeout

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

    def query(self, user_input: str, system_message: str) -> str:
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
        return choices[0].get("message", {}).get("content") or ""

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
