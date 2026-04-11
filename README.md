# CLI Agent Architecture

This repository provides a high-performance AI agent architecture that integrates isolated tool servers with a central orchestrator, focusing on security, reliability, and compatibility. The system utilizes Docker containers to provide a secure environment for executing tool-specific code.

## Prerequisites

* **Node.js (WSL/Linux)**: Node 20+ recommended (install inside WSL; do not rely on Windows `npm.exe`).
* **Docker**: Must be installed and running on your system to manage and run the isolated tool servers.
* **npm**: Used for dependency management/build (or pnpm/yarn if preferred).

## Setup and Installation

### 1. Environment Preparation
* Install dependencies:

```bash
npm install
```

* Create a `.env` file in the project root and add your required API keys (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

### 2. Building Tool Images
The orchestrator requires pre-built Docker images to launch tool servers. For instance, the weather tool requires an image named `weather-server:test`.
* Build the weather tool image:

```bash
npm run build:weather-image
```

## Running the Application

To interact with the agent, you must start the orchestrator server followed by the CLI client.

### 1. Start the Orchestrator
The orchestrator manages the lifecycle of tool containers and handles requests for tool listing, activation, and execution.
* Dev mode:

```bash
npm run dev:orchestrator
```

* Production build + run:

```bash
npm run build
npm run start:orchestrator
```

### 2. Start the CLI Client
In a separate terminal window, start the interactive client to communicate with the agent:
* Dev mode:

```bash
npm run dev:cli
```

* Production build + run:

```bash
npm run build
npm run start:cli
```

## Core Components

* **Orchestrator (`src_ts/orchestrator.ts`)**: A Fastify-based server that manages a registry of tool servers, controls Docker containers, and executes code within those containers.
* **CLI Client (`src_ts/cli_client/cli_client.ts`)**: Manages the conversation history and translates user intent into orchestrator commands.
* **Tool Servers**: Individual tools (like the weather server) are defined within `src/cli_server/` and run in isolated environments to ensure security.

## Project Roadmap

* Add comprehensive tool registration for OpenAI and Anthropic APIs.
* Improve server documentation by including more detailed usage examples in the tool manuals.
