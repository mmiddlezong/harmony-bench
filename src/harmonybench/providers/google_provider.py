"""Google Gemini API (official `google-genai` SDK), AI Studio / Gemini Developer API.

Supported `params` in models.yaml:
  thinking_level:   minimal | low | medium | high   -> ThinkingConfig.thinking_level
  thinking_budget:  int (older 2.5-series models; -1 = dynamic, 0 = off)
  media_resolution: low | medium | high             -> GenerateContentConfig.media_resolution
  extra:            dict merged into GenerateContentConfig as-is
"""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import Provider, ProviderError, ProviderResult, Request, Usage, dump_usage

_REFUSAL_FINISH = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "LANGUAGE",
}


def _enum_name(value) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", None) or getattr(value, "name", None) or str(value)


class GoogleProvider(Provider):
    def __init__(self, spec, client: genai.Client | None = None):
        super().__init__(spec)
        http_options = types.HttpOptions(
            timeout=int(spec.timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=60.0),
        )
        if spec.base_url:
            http_options.base_url = spec.base_url
        self.client = client or genai.Client(api_key=spec.api_key() or "unset", http_options=http_options)

    def build_request(self, request: Request) -> dict:
        p = self.spec.params
        config: dict = {"max_output_tokens": self.spec.max_output_tokens}
        if request.schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = request.schema
        thinking: dict = {}
        if p.get("thinking_level"):
            thinking["thinking_level"] = str(p["thinking_level"]).upper()
        if p.get("thinking_budget") is not None:
            thinking["thinking_budget"] = int(p["thinking_budget"])
        if thinking:
            config["thinking_config"] = thinking
        if p.get("media_resolution"):
            config["media_resolution"] = f"MEDIA_RESOLUTION_{str(p['media_resolution']).upper()}"
        config.update(p.get("extra", {}))
        contents = self._contents(request)
        return {
            "model": self.spec.model,
            "contents": [c.model_dump(exclude_none=True) for c in contents],
            "config": config,
        }

    @staticmethod
    def _contents(request: Request) -> list[types.Content]:
        parts = []
        if request.image is not None:
            parts.append(types.Part.from_bytes(data=request.image, mime_type=request.media_type))
        parts.append(types.Part.from_text(text=request.prompt))
        return [types.Content(role="user", parts=parts)]

    async def complete(self, request: Request) -> ProviderResult:
        req = self.build_request(request)
        try:
            resp = await self.client.aio.models.generate_content(
                model=req["model"],
                contents=self._contents(request),
                config=types.GenerateContentConfig(**req["config"]),
            )
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                raise  # rate limit after SDK retries: retryable at the runner level
            raise ProviderError(f"ClientError {getattr(e, 'code', '')}: {e}") from e

        finish = None
        text_parts: list[str] = []
        if resp.candidates:
            cand = resp.candidates[0]
            finish = _enum_name(cand.finish_reason)
            for part in cand.content.parts if cand.content and cand.content.parts else []:
                if part.text and not part.thought:
                    text_parts.append(part.text)
        block = None
        if resp.prompt_feedback is not None:
            block = _enum_name(resp.prompt_feedback.block_reason)

        um = resp.usage_metadata
        usage = Usage()
        if um is not None:
            thoughts = um.thoughts_token_count or 0
            usage = Usage(
                input_tokens=um.prompt_token_count or 0,
                # Gemini bills thinking tokens as output; candidates_token_count excludes them.
                output_tokens=(um.candidates_token_count or 0) + thoughts,
                reasoning_tokens=thoughts,
                cached_input_tokens=um.cached_content_token_count or 0,
            )
        return ProviderResult(
            text="".join(text_parts),
            usage=usage,
            stop_reason=block or finish,
            request_id=getattr(resp, "response_id", None),
            refused=bool(block) or (finish in _REFUSAL_FINISH),
            truncated=finish == "MAX_TOKENS",
            served_model=getattr(resp, "model_version", None),
            raw_usage=dump_usage(um),
        )

    async def aclose(self) -> None:
        try:
            await self.client.aio.aclose()
        except Exception:
            pass
