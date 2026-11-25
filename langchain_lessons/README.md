# LangChain Insurance Claims Assistant

A hands-on AI Engineering mini-project combining LangChain agents, local LLMs, tool calling, middleware, Pydantic validation and SQLite.

## Features

- Local Qwen2 inference using Ollama
- Provider-independent model configuration
- Read-only insurance claim retrieval
- Agent-driven tool execution
- Model and tool execution limits
- Pydantic-based structured reports
- Independent verification of financial calculations
- Automated tests using pytest

## Technology Stack

Python, LangChain, LangGraph, Ollama, Qwen2, Pydantic, SQLite and pytest.

## Getting Started

From the repository root, activate the existing virtual environment and install the required project dependencies.

Start Ollama and confirm that `qwen2:7b` is available.

Initialize the sample database if needed:

```bash
python -m tool_calling.setup_claims_db
```

Run the assistant:

```bash
python -m langchain_lessons.insurance_assistant
```

## Automated Testing

Run the deterministic tests:

```bash
python -m pytest tests/test_insurance_assistant.py -v
```

These tests use an isolated temporary SQLite database and simulated LangChain messages. They do not require live LLM inference.

## Example Query

Find all approved claims of at least $1,000, calculate their total exposure, and summarize the results.

For the sample dataset, the expected records are CLM1004 and CLM1005, with a combined amount of $5,000.

## Limitations

This is a learning project, not a production insurance application.

It uses fictional claims and a fixed reporting request. The verification process checks financial tool results, but the AI-generated narrative summary is not independently fact-checked. Authentication, tenant-level authorization, detailed audit logging, production error handling and deployment infrastructure are future improvements.

## Future Improvements

- Dynamic user filters validated with Pydantic
- API endpoints using FastAPI
- Human-in-the-loop approvals
- LangSmith tracing
- Regression evaluations for LLM responses
- Containerization and AWS deployment