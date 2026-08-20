import os
import sqlite3
from openai import OpenAI
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel,Field
from typing import Literal
from fastapi import FastAPI


model_image = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("QWEN_API_KEY"),
);

model_embeddingg = init_chat_model(
    model="text-embedding-v3",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("QWEN_API_KEY"),
);

model_chat = init_chat_model(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
);

agent_chat = create_agent(model_chat);
agent_image = create_agent(model_image);

# ── 图片识电影：千问 VL 视觉模型 ──
model_vision = init_chat_model(
    model="qwen3-vl-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("QWEN_API_KEY"),
)
agent_vision = create_agent(model_vision)
