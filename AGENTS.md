# Agent Starter Python - Development Guide

## Project Overview

This is a LiveKit-based voice AI assistant backend with two main components:

1. **Agent Service** (`src/agent.py`) - Voice AI pipeline handling STT, LLM, and TTS
2. **Token Server** (`src/server.py`) - FastAPI server providing LiveKit access tokens

### Tech Stack

- **Framework**: LiveKit Agents SDK (Python)
- **LLM**: OpenAI GPT-4.1-mini
- **STT**: OpenAI gpt-4o-mini-transcribe
- **TTS**: ElevenLabs Multilingual V2
- **VAD**: Silero VAD
- **Turn Detection**: LiveKit Multilingual Model
- **API Server**: FastAPI with Uvicorn
- **Package Manager**: uv

## Setup Instructions

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Copy `.env.example` to `.env.local` and set the following variables:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `ELEVEN_API_KEY`

### 3. Download Required Models (First Time Only)

```bash
uv run python src/agent.py download-files
```

This downloads Silero VAD and LiveKit turn detector models.

## Running the Application

### Start the Agent Service

```bash
uv run python src/agent.py dev
```

### Start the Token Server

Run in a separate terminal:

```bash
uv run python src/server.py
```

The token server runs on `http://localhost:10001` and provides connection details for frontend clients.

## Additional Commands

### Test Agent in Terminal

```bash
uv run python src/agent.py console
```

## Searching Documentation

Always use Context7 MCP to search the documentation before generate the code from the external libraries.

## Important Guidelines

**⚠️ Security Notice**: Never hardcode environment variables, API keys, or secrets in source files. Always use `.env.local` for sensitive configuration.
