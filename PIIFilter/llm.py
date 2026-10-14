"""Chat LLM backends that receive the masked prompt.

local:     an open-weight model run with transformers, standing in for the cloud LLM
openai:    OpenAI Responses API (needs OPENAI_API_KEY)
anthropic: Claude Messages API (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
"""

SYSTEM_PROMPT = (
    "Some personal details in the user's message have been replaced with placeholders "
    "such as <NAME_1> or <PHONE_1>. Treat each placeholder as the real value: wherever "
    "that detail belongs in your answer, copy the placeholder exactly as written, "
    "including the angle brackets."
)


class LocalLLM:
    default_model = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, model: str | None = None, device: str = "cuda", max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model or self.default_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=device
        )
        self.max_new_tokens = max_new_tokens

    def chat(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        # Greedy decoding, so the toy experiment is reproducible.
        output = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAILLM:
    default_model = "gpt-5-mini"

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.model_id = model or self.default_model
        self.client = OpenAI()

    def chat(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        response = self.client.responses.create(model=self.model_id, instructions=system, input=prompt)
        return response.output_text.strip()


class AnthropicLLM:
    default_model = "claude-opus-5-5"
    # Models that accept the server-side refusal fallback ("default" form).
    fallback_models = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

    def __init__(self, model: str | None = None, effort: str | None = "medium"):
        import anthropic

        self.model_id = model or self.default_model
        self.effort = effort
        self.client = anthropic.Anthropic()

    def chat(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        kwargs = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.model_id in self.fallback_models:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        response = self.client.beta.messages.create(
            model=self.model_id,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {response.stop_details}")
        return "".join(b.text for b in response.content if b.type == "text").strip()


BACKENDS = {"local": LocalLLM, "openai": OpenAILLM, "anthropic": AnthropicLLM}


def make_llm(backend: str, model: str | None = None):
    return BACKENDS[backend](model=model)
