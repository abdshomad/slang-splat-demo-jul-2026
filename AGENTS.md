# Agent Rules

When working on this workspace, please refer to the specific guidelines for orchestrating the demo application:

- [Demo Wrapper Agent Guidelines](file:///home/aiserver/LABS/DIGITAL-TWIN-3D/slang-splat-demo-jul-2026/demo-wrapper-only/AGENTS.md)

Follow all instructions in that file regarding Docker Compose, shell script entrypoints (`install.sh`, `start.sh`, `stop.sh`, `monitor.sh`, `restart.sh`), testing screenshots, and issue tracking.

## Python Dependency Management

Always use `uv` for Python environments and dependency operations:
- Use `uv init` to initialize projects.
- Use `uv add` to add packages.
- Use `uv sync` to sync packages.
- Use `uv run` to run scripts.
