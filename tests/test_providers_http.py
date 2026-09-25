"""Provider integration tests against a local mock HTTP server.

These drive the REAL vendor SDKs (anthropic, openai, google-genai) end to end, pointed at
pytest-httpserver instead of the vendor API. They verify the exact wire format each SDK
sends for our requests, and that realistic responses are normalized correctly - without
any API key or spend.
"""

from __future__ import annotations

import base64
import json
import re

import pytest
from werkzeug import Response

from harmonybench.judge import JUDGE_SCHEMA as OUTPUT_SCHEMA
from harmonybench.providers import ProviderError, Request, make_provider

IMAGE = b"\x89PNG\r\n\x1a\nfake-image-bytes"
PROMPT = "What chord is this?"
# Image + schema exercises every request feature at once; the benchmark itself sends no
# schema (free-text answers) and the judge sends no image.
REQ = Request(prompt=PROMPT, image=IMAGE, schema=OUTPUT_SCHEMA, schema_name="chord_judgment")


def _snake(obj):
    """Normalize JSON keys to snake_case. google-genai emits top-level fields in camelCase but
    copies nested typed objects with snake_case keys; the Gemini API accepts both."""
    if isinstance(obj, dict):
        return {re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower(): _snake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snake(v) for v in obj]
    return obj


def _capture(httpserver, path, payload, status=200, headers=None):
    seen: dict = {}

    def handler(request):
        seen["body"] = request.get_json()
        seen["headers"] = dict(request.headers)
        return Response(json.dumps(payload), status=status, content_type="application/json", headers=headers or {})

    httpserver.expect_request(path, method="POST").respond_with_handler(handler)
    return seen


# --------------------------------------------------------------------------- Anthropic
async def test_anthropic(httpserver, spec_factory, good_json):
    spec = spec_factory(provider="anthropic", base_url=httpserver.url_for(""), params={"effort": "high"})
    seen = _capture(
        httpserver,
        "/v1/messages",
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test-model-api-id",
            "content": [{"type": "thinking", "thinking": "", "signature": "x"}, {"type": "text", "text": good_json}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 700,
                "output_tokens": 900,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 600},
            },
        },
        headers={"request-id": "req_123"},
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()

    body = seen["body"]
    assert body["model"] == "test-model-api-id"
    assert body["max_tokens"] == spec.max_output_tokens
    img, txt = body["messages"][0]["content"]
    assert img["type"] == "image" and base64.b64decode(img["source"]["data"]) == IMAGE
    assert txt == {"type": "text", "text": PROMPT}
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}, "effort": "high"}
    assert "thinking" not in body and "temperature" not in body
    assert seen["headers"]["X-Api-Key"] == "sk-test"

    assert result.text == good_json
    assert result.usage.input_tokens == 700 and result.usage.output_tokens == 900
    assert result.usage.reasoning_tokens == 600
    assert result.request_id == "req_123"
    assert not result.refused and not result.truncated


async def test_anthropic_refusal_and_bad_request(httpserver, spec_factory):
    spec = spec_factory(provider="anthropic", base_url=httpserver.url_for(""))
    _capture(
        httpserver,
        "/v1/messages",
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "usage": {"input_tokens": 700, "output_tokens": 3},
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    assert result.refused and result.text == ""

    httpserver.clear()
    _capture(
        httpserver,
        "/v1/messages",
        {"type": "error", "error": {"type": "invalid_request_error", "message": "model not found"}},
        status=400,
    )
    with pytest.raises(ProviderError, match="model not found"):
        await provider.complete(REQ)
    await provider.aclose()


# --------------------------------------------------------------------------- OpenAI Responses
async def test_openai_responses(httpserver, spec_factory, good_json):
    spec = spec_factory(
        provider="openai",
        base_url=httpserver.url_for("/v1"),
        params={"reasoning_effort": "medium", "image_detail": "high"},
    )
    seen = _capture(
        httpserver,
        "/v1/responses",
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": good_json, "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 820,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1500,
                "output_tokens_details": {"reasoning_tokens": 1200},
                "total_tokens": 2320,
            },
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()

    body = seen["body"]
    assert body["model"] == "test-model-api-id" and body["store"] is False
    img, txt = body["input"][0]["content"]
    assert img["type"] == "input_image" and img["detail"] == "high"
    assert img["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(img["image_url"].split(",", 1)[1]) == IMAGE
    assert txt == {"type": "input_text", "text": PROMPT}
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
    assert body["reasoning"] == {"effort": "medium"}
    assert body["max_output_tokens"] == spec.max_output_tokens
    assert seen["headers"]["Authorization"] == "Bearer sk-test"

    assert result.text == good_json
    assert result.usage.output_tokens == 1500 and result.usage.reasoning_tokens == 1200
    assert not result.truncated


async def test_openai_responses_incomplete(httpserver, spec_factory):
    spec = spec_factory(provider="openai", base_url=httpserver.url_for("/v1"))
    _capture(
        httpserver,
        "/v1/responses",
        {
            "id": "resp_2",
            "object": "response",
            "created_at": 0,
            "status": "incomplete",
            "model": "gpt-test",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
            "usage": {
                "input_tokens": 820,
                "output_tokens": 16000,
                "total_tokens": 16820,
                "output_tokens_details": {"reasoning_tokens": 16000},
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()
    assert result.truncated and result.text == ""


async def test_xai_responses_separate_reasoning_tokens(httpserver, spec_factory, good_json):
    """xAI uses the Responses API; if reasoning tokens are reported outside output_tokens
    (but inside total_tokens), they must still be billed as output."""
    spec = spec_factory(provider="xai", base_url=httpserver.url_for("/v1"), params={"reasoning_effort": "high"})
    seen = _capture(
        httpserver,
        "/v1/responses",
        {
            "id": "resp_x",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "grok-test",
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": good_json, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1900,
                "output_tokens": 180,
                "total_tokens": 4080,
                "output_tokens_details": {"reasoning_tokens": 2000},
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()
    assert seen["body"]["reasoning"] == {"effort": "high"} and seen["body"]["store"] is False
    assert result.usage.output_tokens == 2180 and result.usage.reasoning_tokens == 2000
    assert result.raw_usage["total_tokens"] == 4080


# --------------------------------------------------------------------------- Chat Completions (OpenRouter etc.)
@pytest.mark.parametrize("provider_name", ["openrouter", "openai_chat"])
async def test_openai_chat(httpserver, spec_factory, good_json, provider_name):
    spec = spec_factory(
        provider=provider_name,
        base_url=httpserver.url_for("/v1"),
        params={"extra_body": {"provider": {"sort": "price"}}},
    )
    seen = _capture(
        httpserver,
        "/v1/chat/completions",
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 0,
            "model": "vendor/model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": good_json}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 400,
                "total_tokens": 1200,
                "completion_tokens_details": {"reasoning_tokens": 250},
                "cost": 0.0123,
            },
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()

    body = seen["body"]
    img, txt = body["messages"][0]["content"]
    assert img["type"] == "image_url" and img["image_url"]["url"].startswith("data:image/png;base64,")
    assert txt == {"type": "text", "text": PROMPT}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == OUTPUT_SCHEMA
    assert body["provider"] == {"sort": "price"}  # extra_body is merged into the JSON body
    limit_field = "max_tokens" if provider_name == "openrouter" else "max_completion_tokens"
    assert body[limit_field] == spec.max_output_tokens
    assert ("max_completion_tokens" if provider_name == "openrouter" else "max_tokens") not in body

    assert result.text == good_json
    assert result.usage.input_tokens == 800
    assert result.provider_cost_usd == pytest.approx(0.0123)


async def test_openai_chat_length_truncation(httpserver, spec_factory):
    spec = spec_factory(provider="openrouter", base_url=httpserver.url_for("/v1"))
    _capture(
        httpserver,
        "/v1/chat/completions",
        {
            "id": "gen-2",
            "object": "chat.completion",
            "created": 0,
            "model": "vendor/model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": '{"final_'}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 800, "completion_tokens": 16000, "total_tokens": 16800},
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()
    assert result.truncated


# --------------------------------------------------------------------------- Google Gemini
async def test_google(httpserver, spec_factory, good_json):
    spec = spec_factory(
        provider="google",
        model="gemini-test",
        base_url=httpserver.url_for(""),
        params={"thinking_level": "high", "media_resolution": "high"},
    )
    seen = _capture(
        httpserver,
        "/v1beta/models/gemini-test:generateContent",
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "thinking...", "thought": True}, {"text": good_json}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 560,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 800,
                "totalTokenCount": 1560,
            },
            "modelVersion": "gemini-test-001",
            "responseId": "r1",
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()

    body = _snake(seen["body"])
    parts = body["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(parts[0]["inline_data"]["data"]) == IMAGE
    assert parts[1]["text"] == PROMPT
    cfg = body["generation_config"]
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["response_json_schema"] == _snake(OUTPUT_SCHEMA)
    assert cfg["thinking_config"] == {"thinking_level": "HIGH"}
    assert cfg["media_resolution"] == "MEDIA_RESOLUTION_HIGH"
    assert cfg["max_output_tokens"] == spec.max_output_tokens
    assert seen["headers"]["X-Goog-Api-Key"] == "sk-test"

    assert result.text == good_json  # thought parts excluded
    assert result.usage.output_tokens == 1000 and result.usage.reasoning_tokens == 800
    assert result.served_model == "gemini-test-001"


async def test_google_safety_block(httpserver, spec_factory):
    spec = spec_factory(provider="google", model="gemini-test", base_url=httpserver.url_for(""))
    _capture(
        httpserver,
        "/v1beta/models/gemini-test:generateContent",
        {
            "candidates": [{"finishReason": "SAFETY"}],
            "usageMetadata": {"promptTokenCount": 560, "totalTokenCount": 560},
        },
    )
    provider = make_provider(spec)
    result = await provider.complete(REQ)
    await provider.aclose()
    assert result.refused and result.text == ""


# --------------------------------------------------------------------------- request shapes
def test_free_text_and_text_only_requests(spec_factory):
    """Benchmark requests carry no schema; MusicXML-condition and judge requests carry no image."""
    for provider in ("anthropic", "openai", "openrouter", "google"):
        p = make_provider(spec_factory(provider=provider))
        free = p.build_request(Request(prompt=PROMPT, image=IMAGE))
        text_only = p.build_request(Request(prompt=PROMPT))
        blob = str(free)
        assert "json_schema" not in blob and "response_format" not in blob, provider
        assert "base64" not in str(text_only) and "inline_data" not in str(text_only), provider
        assert PROMPT in str(text_only), provider
