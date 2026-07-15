"""Manual live-check script for provider/model configuration."""

from __future__ import annotations

import asyncio
import time

from dotenv import load_dotenv
from openai import APIStatusError

from agents.llm_config import LLMConfig
from agents.multi_agent import build_initial_multi_agent_state, build_multi_agent_graph


def main() -> None:
    load_dotenv()

    provider = LLMConfig.get_provider()
    model_name = LLMConfig.get_model_name()
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")

    topic = "Сколько будет 25 * 4?"
    started_at = time.perf_counter()

    try:
        # Validate credentials and API reachability before agent execution.
        model = LLMConfig.create_chat_model(temperature=0)
        _ = model.invoke("Reply with exactly one token: ok")

        graph = build_multi_agent_graph()
        initial_state = build_initial_multi_agent_state(
            topic=topic,
            user_id=0,
            conversation_history=[],
            use_llm=True,
        )
        result = asyncio.run(graph.ainvoke(initial_state))
        elapsed = time.perf_counter() - started_at

        print(f"Execution time: {elapsed:.2f}s")
        print(f"Revisions: {result['revision_count']}")
        print("Research:", str(result["research_data"])[:300])
        print("Draft:", str(result["draft"])[:400])
    except ValueError as exc:
        print("Configuration error:", exc)
        print("Hint: set OPENAI_API_KEY in ai-agents-lab/.env")
    except APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 402:
            print("Provider returned 402 Insufficient Balance. Top up your account.")
        else:
            print("API status error:", exc)
    except Exception as exc:
        print("Unexpected error while running live check:", exc)


if __name__ == "__main__":
    main()
