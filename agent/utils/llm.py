"""LLM wrapper with token logging and agentic tool-use loop."""
import os
from typing import Callable, Optional

import anthropic

MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

TOOLS = [
    {
        "name": "execute_python",
        "description": (
            "Execute Python code and return stdout and stderr. "
            "pandas, numpy, and scikit-learn are available. "
            "Use absolute paths or paths relative to the project root for file access."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    }
]

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def run_agent_loop(
    messages: list,
    system: str,
    tools: list,
    budget: int,
    log: dict,
    tool_handler: Optional[Callable[[str], dict]] = None,
) -> tuple[str, int]:
    """Run an agentic conversation loop until end_turn or call budget is exhausted.

    Args:
        messages:      Mutable list of message dicts (modified in-place).
        system:        System prompt string.
        tools:         List of tool definitions (pass [] for no tools).
        budget:        Maximum number of LLM API calls allowed.
        log:           Dict to append call records to (must have "llm_calls" and "tool_calls" lists).
        tool_handler:  Callable(code: str) -> {"stdout", "stderr", "success"} for execute_python.

    Returns:
        (final_text, calls_used)
    """
    client = get_client()
    calls = 0
    final_text = ""

    while calls < budget:
        kwargs: dict = {
            "model": MODEL,
            "max_tokens": 2048,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        calls += 1

        log["llm_calls"].append(
            {
                "call_num": len(log["llm_calls"]) + 1,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
            }
        )

        # Append assistant turn (SDK objects are accepted by subsequent API calls)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            break

        if response.stop_reason == "tool_use" and tool_handler:
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    code = block.input.get("code", "")
                    result = tool_handler(code)
                    output = f"stdout:\n{result['stdout']}\nstderr:\n{result['stderr']}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )
                    log["tool_calls"].append(
                        {
                            "code": code[:400],
                            "stdout": result["stdout"][:400],
                            "stderr": result["stderr"][:200],
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool handler or unhandled stop reason — extract any text and stop
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            break

    return final_text, calls
