"""OpenAI-compatible Chat Completions endpoints (OpenRouter, and anything else that speaks
the /chat/completions dialect), via the official `openai` SDK with a base_url.

Supported `params` in models.yaml:
  reasoning_effort: passed as `reasoning_effort` (only for models that accept it)
  image_detail:     low | high | auto (default: high)
  max_tokens_param: name of the output-limit field. Default `max_tokens` for OpenRouter
                    (the only one its endpoints advertise, which matters with
                    `require_parameters: true`), `max_completion_tokens` otherwise
                    (required by OpenAI reasoning models)
  extra_body:       dict sent as extra JSON body fields (e.g. OpenRouter `provider`
                    routing preferences or `reasoning` settings)
  extra:            dict merged into the request kwargs as-is
"""

from __future__ import annotations

import openai

from .base import Provider, ProviderError, ProviderResult, Request, Usage, billable_output, data_url, dump_usage

_FATAL = (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


class OpenAIChatProvider(Provider):
    def __init__(self, spec, client: openai.AsyncOpenAI | None = None):
        super().__init__(spec)
        self.client = client or openai.AsyncOpenAI(
            api_key=spec.api_key() or "unset",  # missing keys fail at request time, not in --dry-run
            base_url=spec.resolved_base_url(),
            timeout=spec.timeout_s,
            max_retries=4,
        )

    def build_request(self, request: Request) -> dict:
        p = self.spec.params
        content: list[dict] = []
        if request.image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url(request.image, request.media_type),
                        "detail": p.get("image_detail", "high"),
                    },
                }
            )
        content.append({"type": "text", "text": request.prompt})
        req: dict = {"model": self.spec.model, "messages": [{"role": "user", "content": content}]}
        default_field = "max_tokens" if self.spec.provider == "openrouter" else "max_completion_tokens"
        req[p.get("max_tokens_param", default_field)] = self.spec.max_output_tokens
        if request.schema is not None:
            req["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "schema": request.schema, "strict": True},
            }
        if p.get("reasoning_effort"):
            req["reasoning_effort"] = p["reasoning_effort"]
        if p.get("extra_body"):
            req["extra_body"] = p["extra_body"]
        req.update(p.get("extra", {}))
        return req

    async def complete(self, request: Request) -> ProviderResult:
        req = self.build_request(request)
        try:
            resp = await self.client.chat.completions.create(**req)
        except _FATAL as e:
            raise ProviderError(f"{type(e).__name__}: {e.message}") from e

        if not resp.choices:
            raise RuntimeError(f"No choices in response: {resp.model_dump_json()[:500]}")
        choice = resp.choices[0]
        msg = choice.message
        finish = choice.finish_reason
        refused = bool(getattr(msg, "refusal", None)) or finish == "content_filter"

        u = resp.usage
        usage = Usage()
        provider_cost = None
        if u is not None:
            out_details = getattr(u, "completion_tokens_details", None)
            in_details = getattr(u, "prompt_tokens_details", None)
            reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
            usage = Usage(
                input_tokens=u.prompt_tokens or 0,
                output_tokens=billable_output(
                    u.completion_tokens or 0, reasoning, u.prompt_tokens or 0, getattr(u, "total_tokens", None)
                ),
                reasoning_tokens=reasoning,
                cached_input_tokens=getattr(in_details, "cached_tokens", 0) or 0,
            )
            # OpenRouter reports the actual charge in usage.cost (USD).
            extra = getattr(u, "model_extra", None) or {}
            if isinstance(extra.get("cost"), (int, float)):
                provider_cost = float(extra["cost"])

        return ProviderResult(
            text=msg.content,
            usage=usage,
            stop_reason=finish,
            request_id=getattr(resp, "_request_id", None) or resp.id,
            refused=refused,
            truncated=finish == "length",
            provider_cost_usd=provider_cost,
            served_model=resp.model,
            raw_usage=dump_usage(u),
        )

    async def aclose(self) -> None:
        await self.client.close()
