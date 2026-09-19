from __future__ import annotations

import os

# Mirrors the structure of src/embeddings.py: one small callable per backend,
# selected by an environment variable, with a mock that keeps the lab runnable
# when no key is configured.
# Alias "-latest" thay vi so phien ban cu the: ban 2.5-flash da bi chan voi key
# moi va lam gay lab, alias thi Google tro sang ban hien hanh.
GEMINI_LLM_MODEL = "gemini-flash-latest"
OPENAI_LLM_MODEL = "gpt-4o-mini"
LLM_PROVIDER_ENV = "LLM_PROVIDER"


class EchoLLM:
    """Stand-in that returns the prompt instead of an answer.

    Not a language model: it exists so KnowledgeBaseAgent can be exercised, and
    so the retrieved context stays inspectable, without any API key.
    """

    def __init__(self) -> None:
        self._backend_name = "echo (khong co LLM that)"

    def __call__(self, prompt: str) -> str:
        return (
            f"Chua cau hinh LLM that (prompt dai {len(prompt)} ky tu). "
            "Dat GEMINI_API_KEY hoac OPENAI_API_KEY trong .env, roi dat LLM_PROVIDER=gemini."
        )


class GeminiLLM:
    """Google Gemini chat backend (google-genai SDK).

    Free tier at aistudio.google.com, which is what the lab recommends for
    students without an OpenAI key.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (hoac GOOGLE_API_KEY) chua duoc dat")

        self.model_name = model_name or os.getenv("GEMINI_LLM_MODEL", GEMINI_LLM_MODEL)
        self._backend_name = f"gemini:{self.model_name}"
        self.client = genai.Client(api_key=api_key)

    def __call__(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return (response.text or "").strip()


class OpenAILLM:
    """OpenAI chat backend."""

    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY chua duoc dat")

        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", OPENAI_LLM_MODEL)
        self._backend_name = f"openai:{self.model_name}"
        self.client = OpenAI()

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()


def make_llm(provider: str | None = None, strict: bool = False):
    """Return an llm_fn for KnowledgeBaseAgent.

    provider defaults to $LLM_PROVIDER, or is inferred from whichever key is
    present. With strict=False a misconfigured backend falls back to EchoLLM so
    the lab keeps running; with strict=True the underlying error is raised,
    which is what a "why is my key not working" check wants.
    """
    provider = (provider or os.getenv(LLM_PROVIDER_ENV) or "").strip().lower()

    if not provider:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif strict:
            raise RuntimeError(
                "Khong tim thay GEMINI_API_KEY hay OPENAI_API_KEY. Dat key trong .env "
                "(xem .env.example), dung dat thang trong ma nguon."
            )
        else:
            return EchoLLM()

    backends = {"gemini": GeminiLLM, "openai": OpenAILLM, "echo": lambda: EchoLLM()}
    if provider not in backends:
        if strict:
            raise ValueError(f"LLM_PROVIDER khong hop le: {provider}")
        return EchoLLM()

    try:
        return backends[provider]()
    except Exception:
        if strict:
            raise
        return EchoLLM()
