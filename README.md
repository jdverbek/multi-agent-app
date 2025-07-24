# Multi-Agent Example App

This directory contains a minimal skeleton for an application that routes tasks
between multiple specialised agents.

## Components

- **MainController**: central orchestrator that receives tasks and dispatches
them to the appropriate agent based on the desired role.
- **AgentGrok4, AgentGPT4o, AgentO3Pro**: example agent classes that would call
external LLM APIs. Each agent exposes an asynchronous `handle()` method.
- **Task**: dataclass representing a unit of work, including the task type,
content payload and optional desired role.

The provided `main.py` script demonstrates creating a controller, submitting a
few example tasks and processing them asynchronously.

## Configuration

Agents expect the relevant API keys to be available as environment variables:

- `OPENAI_API_KEY` for GPT‑4o and O3Pro
- `GROQ_API_KEY` for the Grok‑4 endpoint

Each agent posts the task content to the provider's chat completion endpoint and
returns the generated message as the task response.
