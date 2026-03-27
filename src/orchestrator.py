import inspect
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel
from typing import Optional, Dict, Set

import logging

import requests
import os
import asyncio
from pathlib import Path
import subprocess
import socket

from assets.request_headers import CodeRequest, CommandRequest

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("manual_generator")

# --- 1. The Encapsulated Logic ---
class AIOrchestrator:
    def __init__(self):
        self.registry: Dict[str, Dict[str, str]] = {} # {servername: {summary: server_summary, etc.}}
        self.active_servers: Dict[str, Dict[str, int | subprocess.Popen]] = {} # {servername: {port: process}}
        self.next_port = 8001 # need turn-around logic

        # initialize the registry
        current_dir = Path.cwd()
        docs_path = current_dir / "cli_server/docs"
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            logging.info(f"--- Checking file: {file_path} ---")
            
            # Ensure we are only reading files (skipping subdirectories)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.registry.update({filename.split('.')[0]: {
                        "summary": f.readline().strip(),
                        "cli": f.readline().strip()
                    }})
                    logging.info(f"Server summary and cli path read from {filename}")
            except Exception as e:
                logging.error(f"Could not read {filename}: {e}")
        
        # self.list_servers()
        # asyncio.run(self.activate_server("weather"))
        # asyncio.run(self.stop_server("weather"))

    def list_servers(self) -> Dict[str, str]:
        summary = {name: info["summary"] for name, info in self.registry.items()}
        return str(summary)

    async def activate_server(self, name: str) -> str:
        if name not in self.registry:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=f"Server {name} not found in registry.")
        
        port = self.next_port

        # --- Prepare the environment ---
        env = os.environ.copy()
        # Add the current directory (where the orchestrator is running) to PYTHONPATH
        env["PYTHONPATH"] = os.getcwd() 

        # Pass the env to Popen
        process = subprocess.Popen(
            ["uv", "run", self.registry[name]["cli"], str(port)],
            env=env
        )
        logging.info(f"Activated server: {name} on port {port}")

        # Wait for the server to start
        max_retries = 20
        server_ready = False
        for i in range(max_retries):
            try:
                # Try to open a socket to the port
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    server_ready = True
                    logging.info(f"Server {name} confirmed ready.")
                    break
            except (ConnectionRefusedError, socket.timeout):
                await asyncio.sleep(0.2) # Wait 200ms before retrying
        
        if not server_ready:
            process.terminate()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server failed to start in time")
        
        self.active_servers.update({name: {"port": port, "process": process}})
        self.next_port += 1

        # Read the manual
        current_dir = Path.cwd()
        manual_path = current_dir / f"cli_server/docs/{name}.md"
        manual = ""
        with open(manual_path, "r") as f:
            for _ in range(3):
                next(f, None)
            manual = f.read()
        return manual
    
    async def stop_server(self, name: str):
        if name not in self.registry:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=f"Server {name} not found in registry.")
        
        self.active_servers[name]["process"].terminate()
        logging.info(f"Stopped server: {name} on port {self.active_servers[name]['port']}")
        self.active_servers.pop(name)
        return f"Server {name} stopped"

    async def execute(self, name: str, language: str, code: str):
        if name not in self.active_servers:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Server {name} not activated")
        
        server_url = f"http://127.0.0.1:{self.active_servers[name]['port']}/execute"
        data = {
            "language": language,
            "code": code
        }
        response = requests.post(server_url, json=data)
        # logging.info(f"Raw Response:\n{response.status_code} - {response.text}")
        # logging.info("hihihiha")
        if response.status_code != status.HTTP_200_OK:
            raise HTTPException(status_code=response.status_code, detail=f"Error from server: {response.json()['detail']}")
        return response.json()['stdout']

# --- 2. The API Layer ---
app = FastAPI()
orchestrator = AIOrchestrator()  # Initialize once

@app.post("/orchestrate")
async def handle_request(req: CommandRequest, response: Response) -> str:
    if req.command == "list":
        return orchestrator.list_servers()
    
    elif req.command == "activate":
        return await orchestrator.activate_server(req.server_name)
    
    elif req.command == "stop":
        return await orchestrator.stop_server(req.server_name)
    
    elif req.command == "execute":
        return await orchestrator.execute(req.server_name, req.language, req.code)
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

