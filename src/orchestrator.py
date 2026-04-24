import inspect
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel
from typing import Optional, Dict, Set, Any
from collections import defaultdict

import logging

import requests
import os
import asyncio
from pathlib import Path

import json
import socket
import docker
import httpx

from assets.request_headers import CommandRequest

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("orchestrator")

# --- 1. The Encapsulated Logic ---
class AIOrchestrator:
    def __init__(self):

        self.docker_client = docker.from_env()
        self.registry: Dict[str, Dict[str, str]] = {} # {servername: {summary: server_summary, etc.}}
        self.active_servers: Dict[str, Dict[str, int | Any]] = {} # {servername: {port, container, status, counter}}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.next_port = 8001 # need turn-around logic

        # initialize the registry
        current_dir = Path.cwd()
        docs_path = current_dir / "cli_server/docs"
        # docs_path = current_dir / ".." / "cli_server/docs"
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            logger.info(f"--- Checking file: {file_path} ---")
            
            # Ensure we are only reading files (skipping subdirectories)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.registry.update({filename.split('.')[0]: {
                        "summary": f.readline().strip(),
                        "cli": f.readline().strip()
                    }})
                    logger.info(f"Server summary and cli path read from {filename}")
            except Exception as e:
                logger.error(f"Could not read {filename}: {e}")
        
        # self.list_servers()
        # asyncio.run(self.activate_server("weather"))
        # asyncio.run(self.stop_server("weather"))

    async def list_servers(self) -> Dict[str, str]:
        summary = {name: info["summary"] for name, info in self.registry.items()}
        return {
            "status": status.HTTP_200_OK,
            "result": json.dumps(summary),
            "info": "Fetched Tool Summaries"
        }

    async def activate_server(self, name: str, fetch_manual: bool) -> str:
        if name not in self.registry:
            return {"status": status.HTTP_404_NOT_FOUND, "detail": f"Server {name} not found in registry."}
        
        if name not in self.locks:
            self.locks[name] = asyncio.Lock()
        
        async with self.locks[name]:
            if name in self.active_servers:
                return {"status": status.HTTP_422_UNPROCESSABLE_ENTITY, "detail": f"Server {name} is already active."}
        
        port = self.next_port
        image_tag = f"{name}-server:test"
        container_name = f"{name}-{port}"

        try:
            # Run the container
            # We map the internal port 8000 (from weather_cli.py) to our dynamic host port
            logger.info(f"Starting container: {container_name} on host port {port}...")
            container = self.docker_client.containers.run(
                image_tag,
                detach=True,
                name=container_name,
                ports={'8000/tcp': port}, # {Internal: External}
                remove=True, # Automatically remove container when stopped
                mem_limit="512m",
                cpu_quota=50000,
                network_mode="bridge" # use none if no networking is needed
            )
            
        except Exception as e:
            logger.error(f"Docker Error: {e}")
            return {"status": status.HTTP_500_INTERNAL_SERVER_ERROR, "detail": f"Failed to start Docker container: {str(e)}"}
        
        logger.info(f"Activated server: {name} - {port}")

        # Wait for the API inside the container to become reachable
        if await self._wait_for_server(port):
            # Store instance-specific data
            self.active_servers[name] = {
                "port": port, 
                "container": container,
                "status": "active",
                "counter": 0
            }
            self.next_port += 1 # Ensure the next instance gets a new port
            
            if fetch_manual:
                # Return the manual/docs for the specific server
                current_dir = Path.cwd()
                manual_path = current_dir / f"cli_server/docs/{name}.md"
                manual = ""
                with open(manual_path, "r") as f:
                    for _ in range(2): # TODO: change this magical number
                        next(f, None)
                    manual = f.read()
                return {
                    "status": status.HTTP_200_OK,
                    "result": manual,
                    "info": f"Activated Server: {name} - {port}"
                }
            else:
                return {
                    "status": status.HTTP_200_OK,
                    "result": None,
                    "info": f"Activated Server: {name} - {port}"
                }
        else:
            container.stop()
            return {"status": status.HTTP_500_INTERNAL_SERVER_ERROR, "detail": "Server failed to start."}
    
    async def _wait_for_server(self, port: int, timeout: int = 20) -> bool:
        """Polls the port to see if the FastAPI server is ready via HTTP."""
        url = f"http://127.0.0.1:{port}/openapi.json" # FastAPI automatically serves this
        
        async with httpx.AsyncClient() as client:
            for _ in range(timeout):
                try:
                    # If the server responds at all (even with a 404), it's alive.
                    await client.get(url, timeout=0.5)
                    return True
                except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout):
                    # Server isn't fully booted yet, wait and try again
                    await asyncio.sleep(0.5)
        return False
        
    
    async def stop_server(self, name: str):
        if name not in self.registry:
            return {"status": status.HTTP_404_NOT_FOUND, "detail": f"Server {name} not found in registry."}
        
        # 1. Mark as stopping while holding the lock briefly
        async with self.locks[name]:
            if name not in self.active_servers:
                return {"status": status.HTTP_404_NOT_FOUND, "detail": f"Server {name} already stopped or not active."}
            self.active_servers[name]["status"] = "stopping"

        # 2. WAIT WITHOUT HOLDING THE LOCK
        # This allows existing 'execute' calls to finish and decrement the counter
        max_retries = 50
        while self.active_servers.get(name, {}).get("counter", 0) > 0 and max_retries > 0:
            await asyncio.sleep(0.1)
            max_retries -= 1

        # 3. Final cleanup under lock
        async with self.locks[name]:
            if name in self.active_servers:
                port = self.active_servers[name]['port'] # Save for logging
                container = self.active_servers[name]["container"]
                container.stop()
                self.active_servers.pop(name)
                logger.info(f"Stopped server: {name} on port {port}")
                return {
                    "status": status.HTTP_200_OK,
                    "result": None,
                    "info": f"Stopped Server: {name} - {port}"
                }
            else:
                return {"status": status.HTTP_404_NOT_FOUND, "detail": f"Server {name} already stopped or not active."}
        

    async def execute(self, name: str, language: str, code: str):
        if name not in self.registry:
            return {"status": status.HTTP_404_NOT_FOUND, "detail": f"Server {name} not found in registry."}
        
        async with self.locks[name]:
            if name not in self.active_servers:
                return {"status": status.HTTP_422_UNPROCESSABLE_ENTITY, "detail": f"Server {name} not activated"}
            elif self.active_servers[name]["status"] == "stopping":
                return {"status": status.HTTP_503_SERVICE_UNAVAILABLE, "detail": f"Server {name} is stopping and cannot accept new requests."}
            self.active_servers[name]["counter"] += 1
        
        server_url = f"http://127.0.0.1:{self.active_servers[name]['port']}/execute"
        data = {
            "language": language,
            "code": code
        }
        
        # Use async httpx instead of requests for non-blocking calls
        try:
            async with httpx.AsyncClient() as client:
                # Added a timeout to prevent the orchestrator from hanging
                response = await client.post(server_url, json=data, timeout=10.0)
                if response.status_code != status.HTTP_200_OK:
                    return {"status": response.status_code, "detail": response.json().get('detail')}
                return {
                    "status": status.HTTP_200_OK,
                    "result": response.json()['stdout'].strip(),
                    "info": f"Executed Server: {name} - {self.active_servers[name]['port']}"
                }
        finally:
            async with self.locks[name]:
                self.active_servers[name]["counter"] -= 1


# --- 2. The API Layer ---
app = FastAPI()
_orchestrator: Optional[AIOrchestrator] = None


def get_orchestrator() -> AIOrchestrator:
    """Lazy init so importing this module does not require a running Docker daemon."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AIOrchestrator()
    return _orchestrator


@app.post("/orchestrate")
async def handle_request(req: CommandRequest, response: Response) -> dict:
    print(f"Received request: {req}")
    orch = get_orchestrator()
    if req.command == "list_available_servers":
        result = await orch.list_servers()
    
    elif req.command == "activate_server":
        result = await orch.activate_server(req.server_name, req.fetch_manual)
    
    elif req.command == "stop_server":
        result = await orch.stop_server(req.server_name)
    
    elif req.command == "execute_server_code":
        result = await orch.execute(req.server_name, req.language, req.code)
    
    result["id"] = req.id
    return result
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

