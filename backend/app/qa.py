from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI
from google import genai

from .core.config import settings
from .observability import observation


class AnswerGenerationError(Exception):
    pass


@dataclass
class ContextChunk:
    content: str
    document_title: str
    source: str | None
    product_area: str | None
    release_version: str | None
    heading: str | None
    score: float
    index: int


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise AnswerGenerationError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=settings.openai_api_key)


def _gemini_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise AnswerGenerationError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=settings.gemini_api_key)


def build_prompt(question: str, contexts: Sequence[ContextChunk]) -> str:
    lines: list[str] = []
    lines.append("You are an enterprise support assistant.")
    lines.append(
        "Use only the context sections below to answer the question. "
        "If the context does not contain enough information, say that the "
        "answer cannot be confidently determined from the available documents."
    )
    lines.append("")
    lines.append("Context:")

    for i, ctx in enumerate(contexts, start=1):
        lines.append(f"[{i}] Document: {ctx.document_title}")
        if ctx.source:
            lines.append(f"Source: {ctx.source}")
        if ctx.product_area:
            lines.append(f"Product area: {ctx.product_area}")
        if ctx.release_version:
            lines.append(f"Release version: {ctx.release_version}")
        if ctx.heading:
            lines.append(f"Heading: {ctx.heading}")
        lines.append(f"Score: {ctx.score:.3f}")
        lines.append("---")
        lines.append(ctx.content)
        lines.append("")

    lines.append("Instructions:")
    lines.append("- Answer only using the context above.")
    lines.append(
        "- If the answer is not clearly supported by the context, respond with: "
        "\"The answer cannot be confidently determined from the available documents.\""
    )
    lines.append("- Do not fabricate sources or details.")
    lines.append(
        "- When you use information from a context section, cite it inline as "
        "[doc: <document title>, chunk: <index>]."
    )
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")

    return "\n".join(lines)


def generate_answer(question: str, contexts: Sequence[ContextChunk]) -> str:
    prompt = build_prompt(question, contexts)

    try:
        with observation(
            "answer_generation",
            as_type="generation",
            input={"question": question, "num_context_chunks": len(contexts)},
            metadata={
                "provider": settings.llm_provider,
                "model": settings.gemini_chat_model
                if settings.llm_provider == "gemini"
                else settings.openai_chat_model,
            },
        ) as obs:
            if settings.llm_provider == "gemini":
                client = _gemini_client()
                response = client.models.generate_content(
                    model=settings.gemini_chat_model,
                    contents=prompt,
                )
                content = (getattr(response, "text", None) or "").strip()
            else:
                client = _client()
                response = client.chat.completions.create(
                    model=settings.openai_chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful enterprise support copilot. "
                                "Follow the user's prompt exactly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.1,
                )

                choice = response.choices[0]
                content = (choice.message.content or "").strip()

            if obs is not None:
                obs.update(output=content)

            return content
    except Exception as exc:
        raise AnswerGenerationError("Failed to generate answer.") from exc

