"""Clean streaming generator that only shows progress and final answer."""

import json
import time

from openai.types.chat import ChatCompletionChunk

from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.stream import OpenAIStreamingGenerator


class CleanStreamingGenerator(OpenAIStreamingGenerator):
    """Shows only human-readable progress messages and the final answer.
    Tool calls and raw JSON are hidden."""

    name: str = "open_webui"

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self._step = 0

    def add_chunk(self, chunk: ChatCompletionChunk, phase_id: str):
        """Suppress raw LLM streaming chunks (thinking tokens, tool call JSON)."""
        pass

    def add_tool_call(self, phase_id: str, tool: BaseTool) -> None:
        """Show a brief one-line status for each tool call."""
        self._step += 1
        tool_name = tool.tool_name

        if tool_name == "vkusvill_products_search":
            q = getattr(tool, "q", "")
            msg = f"🔍 Ищу: **{q}**...\n\n"
        elif tool_name == "vkusvill_batch_search":
            queries = getattr(tool, "queries", [])
            msg = f"🔍 Ищу {len(queries)} товаров: {', '.join(queries)}...\n\n"
        elif tool_name == "vkusvill_cart_link_create":
            count = len(getattr(tool, "products", []))
            msg = f"🛒 Создаю корзину ({count} товаров)...\n\n"
        elif tool_name == "vkusvill_product_details":
            msg = f"📋 Загружаю детали товара...\n\n"
        elif tool_name == "finalanswertool":
            # Final answer will be shown via add_tool_result
            return
        else:
            msg = f"⚙️ {tool_name}...\n\n"

        self.add_content_delta(msg, phase_id)

    def add_tool_result(self, phase_id: str, content: str, tool_name: str | None = None):
        """Show brief result for search, full result for final answer."""
        if tool_name == "vkusvill_batch_search":
            try:
                items = json.loads(content)
                found = [i for i in items if i.get("found")]
                not_found = [i for i in items if not i.get("found")]
                parts = []
                for i in found:
                    parts.append(f"✅ {i['name']} — {i.get('price', '?')} ₽")
                for i in not_found:
                    parts.append(f"⚠️ {i['query']} — не найден")
                msg = "\n".join(parts) + "\n\n"
            except (json.JSONDecodeError, KeyError):
                msg = ""
            if msg:
                self.add_content_delta(msg, phase_id)

        elif tool_name == "vkusvill_products_search":
            # Extract first product name from result
            try:
                data = json.loads(content)
                if data.get("ok"):
                    items = data["data"].get("items", [])
                    if items:
                        name = items[0].get("name", "").replace("&nbsp;", " ")
                        price = items[0].get("price", {}).get("current", "")
                        msg = f"✅ Нашёл: {name} — {price} ₽\n\n"
                    else:
                        msg = "⚠️ Товар не найден\n\n"
                else:
                    msg = ""
            except (json.JSONDecodeError, KeyError):
                msg = ""
            if msg:
                self.add_content_delta(msg, phase_id)

        elif tool_name == "vkusvill_cart_link_create":
            try:
                data = json.loads(content)
                if data.get("ok"):
                    link = data["data"].get("link", "")
                    msg = f"✅ Корзина готова!\n\n"
                    self.add_content_delta(msg, phase_id)
                else:
                    self.add_content_delta("⚠️ Ошибка создания корзины, пробую снова...\n\n", phase_id)
            except (json.JSONDecodeError, KeyError):
                pass

        elif tool_name == "finalanswertool":
            # This is the final answer — show it fully
            self.add_content_delta(f"\n---\n\n{content}\n", phase_id)

        # Other tool results — skip silently

    def finish(self, phase_id: str, content: str | None = None, finish_reason: str = "stop"):
        """Suppress duplicate final content — already shown via add_tool_result."""
        super().finish(phase_id, content=None, finish_reason=finish_reason)
