# CLI Agent Architecture

This repository provides a high-performance AI agent architecture that integrates isolated tool servers with a central orchestrator, focusing on security, reliability, and compatibility. The system utilizes Docker containers to provide a secure environment for executing tool-specific code.

## Prerequisites

* **Python**: Version 3.10 or higher is required.
* **Docker**: Must be installed and running on your system to manage and run the isolated tool servers.
* **uv**: This project uses `uv` for fast and reliable Python package management.

## Setup and Installation

### 1. Environment Preparation
* Initialize the project environment and install dependencies using the following command:
    ```bash
    uv sync
    ```
    This installs necessary libraries including `fastapi`, `docker`, `httpx`, and SDKs for various language models such as `anthropic`, `openai`, and `google-genai`.
* Create a `.env` file in the project root and add your required API keys (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`).

### 2. Building Tool Images
The orchestrator requires pre-built Docker images to launch tool servers. For instance, the weather tool requires an image named `weather-server:test`.
* Run the image builder script to generate the necessary Docker images:
    ```bash
    uv run src/cli_server/docker_image_builder.py
    ```
    This script builds the `weather-server` image and saves it locally for the orchestrator to access.

## Running the Application

To interact with the agent, you must start the orchestrator server followed by the CLI client.

### 1. Start the Orchestrator
The orchestrator manages the lifecycle of tool containers and handles requests for tool listing, activation, and execution.
* Launch the orchestrator server:
    ```bash
    uv run src/orchestrator.py
    ```
    By default, the orchestrator runs a FastAPI application on `http://127.0.0.1:8000`.

### 2. Start the CLI Client
In a separate terminal window, start the interactive client to communicate with the agent:
* Launch the CLI client:
    ```bash
    uv run src/cli_client/cli_client.py
    ```
    The client handles the user interaction loop and uses the language model to determine which tool commands (such as `list`, `activate`, or `execute`) to send to the orchestrator.

## Core Components

* **Orchestrator (`src/orchestrator.py`)**: A FastAPI-based server that manages a registry of tool servers, controls Docker containers, and executes code within those containers.
* **CLI Client (`src/cli_client/cli_client.py`)**: Manages the conversation history and translates user intent into orchestrator commands.
* **Docker Image Builder (`src/cli_server/docker_image_builder.py`)**: A utility to automate the creation of tool-specific Docker images.
* **Tool Servers**: Individual tools (like the weather server) are defined within `src/cli_server/` and run in isolated environments to ensure security.

## Project Roadmap

* Rewrite the architecture in TypeScript for improved performance.
* Add comprehensive tool registration for OpenAI and Anthropic APIs.
* Improve server documentation by including more detailed usage examples in the tool manuals.
