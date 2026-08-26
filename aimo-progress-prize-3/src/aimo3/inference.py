"""vLLM inference engine for AIMO3.

Provides a unified interface for batched LLM inference with:
- H100-optimized vLLM configuration
- Prefix caching for shared system prompts
- Configurable sampling (temperature, top_p, n_samples)
- Chat template support
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    """Configuration for the vLLM inference engine."""

    model_path: str = "nvidia/OpenMath-Nemotron-14B-Kaggle"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    dtype: str = "bfloat16"
    enable_prefix_caching: bool = True
    max_num_seqs: int = 64
    trust_remote_code: bool = True
    # Sampling defaults
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    n_samples: int = 32
    stop_sequences: list[str] = field(
        default_factory=lambda: ["```output"]
    )


class InferenceEngine:
    """vLLM-based inference engine for math reasoning.

    Lazily initializes the model on first call, so import is fast.
    """

    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self._llm = None
        self._tokenizer = None

    @property
    def llm(self):
        """Lazy-load vLLM model."""
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                enable_prefix_caching=self.config.enable_prefix_caching,
                max_num_seqs=self.config.max_num_seqs,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._tokenizer = self._llm.get_tokenizer()
        return self._llm

    @property
    def tokenizer(self):
        """Get the tokenizer (loads model if needed)."""
        if self._tokenizer is None:
            _ = self.llm  # trigger lazy load
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        n_samples: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of formatted prompt strings.
            n_samples: Number of completions per prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per completion.
            stop: Stop sequences.

        Returns:
            List of lists of completion strings, one list per prompt.
        """
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens or self.config.max_tokens,
            n=n_samples or self.config.n_samples,
            stop=stop or self.config.stop_sequences,
        )

        outputs = self.llm.generate(prompts, params, use_tqdm=False)

        results = []
        for output in outputs:
            completions = [o.text for o in output.outputs]
            results.append(completions)
        return results

    def generate_single(
        self,
        prompt: str,
        n_samples: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate completions for a single prompt.

        Convenience wrapper around generate() for one prompt.
        """
        results = self.generate(
            [prompt],
            n_samples=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return results[0]

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        n_samples: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate completions from chat-format messages.

        Applies the model's chat template before generating.

        Args:
            messages: List of {'role': ..., 'content': ...} dicts.
            n_samples: Number of completions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per completion.
            stop: Stop sequences.

        Returns:
            List of completion strings.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate_single(
            prompt,
            n_samples=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def continue_generation(
        self,
        prompt_with_partial: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Continue a single generation from a partial output.

        Used in TIR loops where we append code output and continue.
        Returns a single greedy completion.
        """
        results = self.generate_single(
            prompt_with_partial,
            n_samples=1,
            temperature=0.0,  # greedy for continuation
            max_tokens=max_tokens or self.config.max_tokens,
            stop=stop or self.config.stop_sequences,
        )
        return results[0] if results else ""
