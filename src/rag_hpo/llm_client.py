from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import date

from rag_hpo._optional import require

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_JSON_HINT = "\n\nRespond with a single valid JSON object and nothing else."


def _strip_think(text: str) -> str:
    """Reasoning models may prepend a <think> block; its braces confuse JSON extraction."""
    return _THINK_BLOCK.sub("", text).strip()


def _extract_json_object(text: str) -> str:
    """Return the outermost {...} if it parses as JSON, otherwise the text unchanged."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return text


class TokenBudgetExceeded(RuntimeError):
    pass


class ILLMClient(ABC):
    @abstractmethod
    def query(self, user_input: str, system_message: str) -> str:
        """Abstract method to query the LLM for user_input."""
        ...

    def query_batch(self, user_inputs: list[str], system_message: str) -> list[str]:
        """Default: one query per input. Subclasses can override with real batching."""
        return [self.query(u, system_message) for u in user_inputs]


class LLMClient(ILLMClient):
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
        import requests
        import tiktoken

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
        return _strip_think(content)

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


class LocalLLMClient(ILLMClient):
    """Runs a Hugging Face chat model inside the current Python process."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        json_mode: bool = True,
        enable_thinking: bool = False,
        batch_size: int = 8,
        device_map: str = "auto",
        dtype: str = "auto",
    ) -> None:
        torch = require("torch", "local-llm")
        transformers = require("transformers", "local-llm")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.json_mode = json_mode
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"  # required for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        self.model.eval()

    def query(self, user_input: str, system_message: str) -> str:
        """Queries the LLM"""
        return self._generate([user_input], system_message)[0]

    def query_batch(self, user_inputs: list[str], system_message: str) -> list[str]:
        """Same as query() for many inputs at once; much faster on a GPU."""
        results: list[str] = []
        for i in range(0, len(user_inputs), self.batch_size):
            results.extend(
                self._generate(user_inputs[i : i + self.batch_size], system_message)
            )
        return results

    def _build_prompt(self, user_input: str, system_message: str) -> str:
        if self.json_mode:
            system_message = system_message + _JSON_HINT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]
        # enable_thinking is used by Qwen3 templates and ignored by others.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _generate(self, user_inputs: list[str], system_message: str) -> list[str]:
        prompts = [self._build_prompt(u, system_message) for u in user_inputs]
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        with self._torch.inference_mode():
            out = self.model.generate(**batch, **gen_kwargs)

        new_tokens = out[:, batch["input_ids"].shape[1] :]  # drop the prompt
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        results = [_strip_think(t) for t in texts]
        if self.json_mode:
            # No server-side JSON mode here, so trim any chatter around the object.
            results = [_extract_json_object(t) for t in results]
        return results
