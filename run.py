#!/usr/bin/env python3
"""VkusVill Shopping Cart Agent."""

import asyncio
import json
import logging
import sys
import os
from typing import ClassVar

# Add sgr-agent-core and agent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sgr-agent-core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.agent_config import GlobalConfig
from sgr_agent_core.agent_factory import AgentFactory
from sgr_agent_core.models import AgentStatesEnum

# Import custom agent so it registers in AgentRegistry
import agent as _agent_module  # noqa: F401

logger = logging.getLogger(__name__)

# ── VkusVill MCP session manager (handles initialize + session-id) ──

MCP_URL = "https://mcp001.vkusvill.ru/mcp"


class VkusvillMCPSession:
    """Manages MCP session with VkusVill server."""

    _session_id: str | None = None
    _initialized: bool = False
    _lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def _http_post(cls, url, json_data, headers, retries=3):
        """HTTP POST with retry logic for flaky connections."""
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(90.0, connect=60.0)
                ) as client:
                    return await client.post(url, json=json_data, headers=headers)
            except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"MCP connection attempt {attempt+1} failed, retrying...")
                await asyncio.sleep(2)

    @classmethod
    async def ensure_initialized(cls):
        if cls._initialized:
            return
        async with cls._get_lock():
            if cls._initialized:
                return
            base_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            # Use a single client for both init + notification (same TCP connection)
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=60.0)) as client:
                        # Step 1: Initialize
                        init_payload = {
                            "jsonrpc": "2.0",
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {"name": "vkusvill-cart-agent", "version": "1.0"},
                            },
                            "id": 1,
                        }
                        r = await client.post(MCP_URL, json=init_payload, headers=base_headers)
                        logger.info(f"MCP init response: {r.status_code}")
                        cls._session_id = r.headers.get("mcp-session-id")
                        logger.info(f"MCP session-id: {cls._session_id}")

                        # Step 2: Send initialized notification (same connection)
                        if cls._session_id:
                            notif_payload = {
                                "jsonrpc": "2.0",
                                "method": "notifications/initialized",
                            }
                            sess_headers = {**base_headers, "Mcp-Session-Id": cls._session_id}
                            r2 = await client.post(MCP_URL, json=notif_payload, headers=sess_headers)
                            logger.info(f"MCP notification response: {r2.status_code}")

                        cls._initialized = True
                        return
                except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                    logger.warning(f"MCP init attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(2)

            # If all retries failed, do NOT mark as initialized
            logger.error("MCP initialization failed after 3 attempts")

    @classmethod
    async def _reinitialize(cls):
        """Force re-initialize MCP session."""
        async with cls._get_lock():
            cls._initialized = False
            cls._session_id = None
        await cls.ensure_initialized()

    @classmethod
    async def call_tool(cls, tool_name: str, arguments: dict) -> str:
        await cls.ensure_initialized()

        for attempt in range(2):
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 2,
            }
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            if cls._session_id:
                headers["Mcp-Session-Id"] = cls._session_id

            r = await cls._http_post(MCP_URL, payload, headers)
            data = r.json()

            # Check for session errors and reinitialize
            if "error" in data:
                err_msg = data["error"].get("message", "")
                if "not initialized" in err_msg.lower() or "session" in err_msg.lower():
                    if attempt == 0:
                        logger.warning("MCP session expired, reinitializing...")
                        await cls._reinitialize()
                        continue
                return f"Error: {json.dumps(data['error'], ensure_ascii=False)}"

            if "result" in data:
                content = data["result"].get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                result = "\n".join(texts) if texts else json.dumps(data["result"], ensure_ascii=False)
                return result

            return json.dumps(data, ensure_ascii=False)

        return "Error: MCP session could not be established"

    @classmethod
    async def call_tool_raw(cls, tool_name: str, arguments: dict) -> str:
        """Like call_tool but returns full untruncated response."""
        await cls.ensure_initialized()

        for attempt in range(2):
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 2,
            }
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            if cls._session_id:
                headers["Mcp-Session-Id"] = cls._session_id

            r = await cls._http_post(MCP_URL, payload, headers)
            data = r.json()

            if "error" in data:
                err_msg = data["error"].get("message", "")
                if "not initialized" in err_msg.lower() or "session" in err_msg.lower():
                    if attempt == 0:
                        logger.warning("MCP session expired, reinitializing...")
                        await cls._reinitialize()
                        continue
                return json.dumps({"ok": False, "error": err_msg}, ensure_ascii=False)

            if "result" in data:
                content = data["result"].get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else json.dumps(data["result"], ensure_ascii=False)

            return json.dumps(data, ensure_ascii=False)

        return json.dumps({"ok": False, "error": "session failed"}, ensure_ascii=False)


# ── VkusVill tool definitions ──


class VkusvillProductsSearch(BaseTool):
    """Поиск товаров ВкусВилл по ключевому слову. Возвращает список товаров с id, xml_id, названием, ценой, рейтингом, весом."""

    tool_name: ClassVar[str] = "vkusvill_products_search"
    q: str = Field(description="Поисковый запрос (например: свёкла, картофель, говядина)")
    page: int = Field(default=1, description="Номер страницы результатов")
    sort: str = Field(default="popularity", description="Сортировка: popularity, rating, price_asc, price_desc")

    async def __call__(self, context, config, **kwargs) -> str:
        args = {"q": self.q}
        if self.page != 1:
            args["page"] = self.page
        if self.sort != "popularity":
            args["sort"] = self.sort
        return await VkusvillMCPSession.call_tool("vkusvill_products_search", args)


class VkusvillBatchSearch(BaseTool):
    """Параллельный поиск нескольких товаров ВкусВилл за один вызов. Принимает список поисковых запросов и возвращает лучший результат по каждому. Используй этот инструмент когда нужно найти несколько ингредиентов сразу."""

    tool_name: ClassVar[str] = "vkusvill_batch_search"
    queries: list[str] = Field(
        description='Список поисковых запросов, например: ["свёкла", "морковь", "говядина", "картофель"]'
    )

    async def __call__(self, context, config, **kwargs) -> str:
        async def search_one(q: str) -> dict:
            raw = await VkusvillMCPSession.call_tool_raw("vkusvill_products_search", {"q": q})
            try:
                data = json.loads(raw)
                if data.get("ok") and data["data"].get("items"):
                    item = data["data"]["items"][0]
                    return {
                        "query": q,
                        "found": True,
                        "xml_id": item["xml_id"],
                        "name": item.get("name", "").replace("&nbsp;", " "),
                        "price": item.get("price", {}).get("current"),
                        "weight": item.get("weight", ""),
                        "rating": item.get("rating", {}).get("average"),
                    }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse search result for '{q}': {e}")
            return {"query": q, "found": False}

        results = await asyncio.gather(*[search_one(q) for q in self.queries])
        return json.dumps(results, ensure_ascii=False)


class VkusvillProductDetails(BaseTool):
    """Получение детальной информации о товаре ВкусВилл по id: состав, КБЖУ, нутриенты."""

    tool_name: ClassVar[str] = "vkusvill_product_details"
    id: int = Field(description="ID товара из результатов поиска")

    async def __call__(self, context, config, **kwargs) -> str:
        return await VkusvillMCPSession.call_tool("vkusvill_product_details", {"id": self.id})


class CartProduct(BaseModel):
    xml_id: int = Field(description="ID товара (xml_id из результатов поиска)")
    q: int = Field(default=1, description="Количество")


class VkusvillCartLinkCreate(BaseTool):
    """Создание ссылки на корзину ВкусВилл. Принимает массив товаров с xml_id и количеством q. Максимум 20 товаров."""

    tool_name: ClassVar[str] = "vkusvill_cart_link_create"
    products: list[CartProduct] = Field(
        description='Массив товаров для корзины.'
    )

    async def __call__(self, context, config, **kwargs) -> str:
        fixed_products = [{"xml_id": int(p.xml_id), "q": p.q} for p in self.products]
        return await VkusvillMCPSession.call_tool("vkusvill_cart_link_create", {"products": fixed_products})


# ── Patch MCP2ToolConverter to skip (we handle MCP manually) ──

import sgr_agent_core.services.mcp_service as mcp_mod


@classmethod
async def _no_mcp_build(cls, config):
    return []


mcp_mod.MCP2ToolConverter.build_tools_from_mcp = _no_mcp_build


# ── Patch OpenAI client timeout ──

_orig_create_client = AgentFactory._create_client


@classmethod
def _patched_create_client(cls, llm_config):
    client = _orig_create_client.__func__(cls, llm_config)
    # Increase timeout for slow models on OpenRouter free tier
    client.timeout = httpx.Timeout(300.0, connect=60.0)
    client.max_retries = 5
    return client


AgentFactory._create_client = _patched_create_client

# ── Patch LLMConfig to pass extra_body correctly ──

from sgr_agent_core.agent_definition import LLMConfig

_orig_to_kwargs = LLMConfig.to_openai_client_kwargs


def _patched_to_kwargs(self) -> dict:
    result = _orig_to_kwargs(self)
    # Remove extra_body from top-level and pass it correctly
    extra_body = result.pop("extra_body", None)
    if extra_body:
        result["extra_body"] = extra_body
    return result


LLMConfig.to_openai_client_kwargs = _patched_to_kwargs


# ── Agent run logic ──


async def run_agent(agent):
    """Run agent with clarification support."""
    execution_task = asyncio.create_task(agent.execute())

    while not execution_task.done():
        if agent._context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
            if agent._context.execution_result:
                print("\n" + agent._context.execution_result + "\n")
            try:
                user_input = input("Ваш ответ: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nОтменено.")
                await agent.cancel()
                return None
            if user_input:
                await agent.provide_clarification([{"role": "user", "content": user_input}])
            else:
                await agent.cancel()
                return None
        await asyncio.sleep(0.1)

    try:
        return await execution_task
    except (asyncio.CancelledError, Exception) as e:
        if not isinstance(e, asyncio.CancelledError):
            print(f"Ошибка: {e}")
        return None


async def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = GlobalConfig.from_yaml(config_path)

    agent_def = config.agents.get("vkusvill_cart")
    if not agent_def:
        print("Агент vkusvill_cart не найден в конфиге")
        sys.exit(1)

    # Inject VkusVill tools
    agent_def.tools.extend([
        VkusvillProductsSearch,
        VkusvillBatchSearch,
        VkusvillProductDetails,
        VkusvillCartLinkCreate,
    ])

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Что хотите приготовить? (или введите список продуктов)")
        query = input("> ").strip()
        if not query:
            return

    print(f"\nСобираю корзину: {query}\n")

    # Pre-initialize MCP session
    await VkusvillMCPSession.ensure_initialized()

    task_messages = [{"role": "user", "content": query}]
    agent = await AgentFactory.create(agent_def, task_messages)

    result = await run_agent(agent)
    if result:
        print(f"\n{result}")
    else:
        print("\nНе удалось получить результат.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nВыход.")
