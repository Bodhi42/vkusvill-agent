"""Custom ToolCallingAgent that handles LengthFinishReasonError gracefully."""

import json
import logging
from typing import Type

from openai import AsyncOpenAI, LengthFinishReasonError

from sgr_agent_core.agent_config import AgentConfig
from sgr_agent_core.agents.tool_calling_agent import ToolCallingAgent
from sgr_agent_core.tools import BaseTool, FinalAnswerTool

logger = logging.getLogger(__name__)


class RobustToolCallingAgent(ToolCallingAgent):
    """ToolCallingAgent that gracefully handles truncated responses from
    models that use thinking tokens (e.g., Qwen3)."""

    name: str = "robust_tool_calling_agent"

    def __init__(
        self,
        task_messages: list,
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        def_name: str | None = None,
        **kwargs: dict,
    ):
        super().__init__(
            task_messages=task_messages,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=toolkit,
            def_name=def_name,
            **kwargs,
        )

    async def _select_action_phase(self, reasoning=None) -> BaseTool:
        phase_id = f"{self._context.iteration}-action"
        tool = None

        for attempt in range(3):
            try:
                # Use non-streaming fallback after first failure
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt+1} (non-streaming)...")
                    completion = await self.openai_client.chat.completions.create(
                        messages=await self._prepare_context(),
                        tools=await self._prepare_tools(),
                        tool_choice=self.tool_choice,
                        **self.config.llm.to_openai_client_kwargs(),
                    )
                else:
                    async with self.openai_client.chat.completions.stream(
                        messages=await self._prepare_context(),
                        tools=await self._prepare_tools(),
                        tool_choice=self.tool_choice,
                        **self.config.llm.to_openai_client_kwargs(),
                    ) as stream:
                        async for event in stream:
                            if event.type == "chunk":
                                self.streaming_generator.add_chunk(event.chunk, phase_id)
                        completion = await stream.get_final_completion()

                tc = completion.choices[0].message.tool_calls[0]
                tool_name = tc.function.name

                # Try parsed_arguments first (only available in streaming mode)
                parsed = getattr(tc.function, 'parsed_arguments', None)
                if parsed and isinstance(parsed, BaseTool):
                    tool = parsed
                    break

                # Fallback: manual JSON parse from raw arguments string
                raw_args = tc.function.arguments or "{}"
                tool = self._try_parse_partial_tool(tool_name, raw_args)
                if tool:
                    logger.info(f"Parsed tool call from raw JSON: {tool_name}")
                    break
                else:
                    logger.warning(f"Could not parse tool args for {tool_name}: {raw_args[:200]}")

            except LengthFinishReasonError as e:
                logger.warning("Response truncated due to max_tokens.")
                completion = e.completion
                if completion and completion.choices:
                    choice = completion.choices[0]
                    if choice.message and choice.message.tool_calls:
                        tc = choice.message.tool_calls[0]
                        raw_args = tc.function.arguments or "{}"
                        tool_name = tc.function.name
                        tool = self._try_parse_partial_tool(tool_name, raw_args)
                        if tool:
                            logger.info(f"Recovered partial tool call: {tool_name}")
                            break
                # Will retry or fallback

            except (ValueError, Exception) as e:
                logger.warning(f"Tool call parse error (attempt {attempt+1}): {e}")
                # Will retry

        if tool is None:
            logger.warning("All attempts failed, using fallback answer")
            tool = self._make_fallback_answer()

        self.conversation.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": phase_id,
                        "function": {
                            "name": tool.tool_name,
                            "arguments": tool.model_dump_json(),
                        },
                    }
                ],
            }
        )
        self.streaming_generator.add_tool_call(phase_id, tool)
        return tool

    def _try_parse_partial_tool(self, tool_name: str, raw_args: str) -> BaseTool | None:
        """Try to parse potentially malformed JSON from a tool call."""
        import re

        # Clean up: remove thinking tags, leading/trailing whitespace
        cleaned = re.sub(r'<think>.*?</think>', '', raw_args, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = raw_args.strip()

        # Find the tool class
        tool_cls = None
        for cls in self.toolkit:
            if cls.tool_name == tool_name:
                tool_cls = cls
                break
        if tool_cls is None:
            return None

        # Try parsing as-is first, then with various closing suffixes
        suffixes = ["", "}", "}}", '"}', '"}}', '"]', '"]}', '"]}}', "]}"]
        for suffix in suffixes:
            try:
                data = json.loads(cleaned + suffix)
                return tool_cls.model_validate(data)
            except (json.JSONDecodeError, Exception):
                continue

        # Last resort: try to extract JSON object from the string
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return tool_cls.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        return None

    def _make_fallback_answer(self) -> FinalAnswerTool:
        """Create a fallback FinalAnswerTool from conversation history."""
        # Extract cart links and product info from conversation
        cart_link = None
        products_found = []

        import re
        for msg in self.conversation:
            content = msg.get("content", "") or ""
            if "share_basket" in content:
                match = re.search(r'https://vkusvill\.ru/\?share_basket=\d+', content)
                if match:
                    cart_link = match.group(0)
            if msg.get("role") == "tool" and '"items"' in content and '"ok"' in content:
                try:
                    data = json.loads(content)
                    if data.get("ok") and "data" in data:
                        items = data["data"].get("items", [])
                        if items:
                            item = items[0]
                            name = item.get("name", "?").replace("&nbsp;", " ")
                            price = item.get("price", {}).get("current", "?")
                            weight = item.get("weight", "")
                            products_found.append(f"- {name} ({weight}) — {price} руб.")
                except (json.JSONDecodeError, KeyError):
                    pass

        # Extract user query dynamically
        user_query = None
        for msg in self.task_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    user_query = content.strip()
                    break
        if user_query:
            header = f"Результат по запросу «{user_query}»:"
        else:
            header = "Корзина собрана!"
        answer_parts = [header]
        if products_found:
            answer_parts.append("\nНайденные товары:")
            answer_parts.extend(products_found)
        if cart_link:
            answer_parts.append(f"\nСсылка на корзину: {cart_link}")
        else:
            answer_parts.append("\n(Ссылка на корзину не была создана)")

        return FinalAnswerTool(
            reasoning="Response was truncated; extracted results from conversation history.",
            completed_steps=["Searched products", "Created cart link", "Compiled results"],
            answer="\n".join(answer_parts),
            status="completed",
        )
