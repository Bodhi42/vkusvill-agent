#!/usr/bin/env python3
"""VkusVill Agent — SGR API server with custom tools and patches."""

import asyncio
import logging
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sgr-agent-core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx

# ── Apply patches BEFORE importing the server ──

# 1. Patch MCP2ToolConverter to skip (we register tools manually)
import sgr_agent_core.services.mcp_service as mcp_mod


@classmethod
async def _no_mcp_build(cls, config):
    return []


mcp_mod.MCP2ToolConverter.build_tools_from_mcp = _no_mcp_build

# 2. Patch AgentFactory client timeout
from sgr_agent_core.agent_factory import AgentFactory

_orig_create_client = AgentFactory._create_client


@classmethod
def _patched_create_client(cls, llm_config):
    client = _orig_create_client.__func__(cls, llm_config)
    client.timeout = httpx.Timeout(300.0, connect=60.0)
    client.max_retries = 5
    return client


AgentFactory._create_client = _patched_create_client

# 3. Patch LLMConfig for extra_body passthrough
from sgr_agent_core.agent_definition import LLMConfig

_orig_to_kwargs = LLMConfig.to_openai_client_kwargs


def _patched_to_kwargs(self):
    result = _orig_to_kwargs(self)
    extra_body = result.pop("extra_body", None)
    if extra_body:
        result["extra_body"] = extra_body
    return result


LLMConfig.to_openai_client_kwargs = _patched_to_kwargs

# 4. Import custom agent, tools, and streaming generator so they register
import agent as _agent_module  # noqa: F401 — registers RobustToolCallingAgent
from clean_stream import CleanStreamingGenerator
from sgr_agent_core.services.registry import StreamingGeneratorRegistry
# Override open_webui with our clean generator
StreamingGeneratorRegistry._items["open_webui"] = CleanStreamingGenerator
StreamingGeneratorRegistry._items["cleanstreaminggenerator"] = CleanStreamingGenerator

# 5. Import VkusVill tools from run.py
from run import (  # noqa: F401
    VkusvillProductsSearch,
    VkusvillBatchSearch,
    VkusvillProductDetails,
    VkusvillCartLinkCreate,
    VkusvillMCPSession,
)

# ── Now load config and start server ──

from sgr_agent_core.agent_config import GlobalConfig
from sgr_agent_core.server.app import app

import uvicorn


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = GlobalConfig.from_yaml(config_path)

    # Inject VkusVill tools into agent definition
    agent_def = config.agents.get("vkusvill_cart")
    if agent_def:
        agent_def.tools.extend([
            VkusvillProductsSearch,
            VkusvillBatchSearch,
            VkusvillProductDetails,
            VkusvillCartLinkCreate,
        ])

    # Initialize MCP session before starting
    asyncio.run(VkusvillMCPSession.ensure_initialized())

    print("Starting SGR server with VkusVill agent on http://localhost:8010")
    print("Frontend: http://localhost:5173")
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")


if __name__ == "__main__":
    main()
