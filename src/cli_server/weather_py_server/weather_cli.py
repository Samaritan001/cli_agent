import logging

import subprocess
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel

import os
import ast
import re

import sys
from pathlib import Path
import asyncio

class CodeRequest(BaseModel):
    language: str
    code: str

app = FastAPI()

@app.post("/execute")
async def execute_code(request: CodeRequest, response: Response):
    # Mapping languages to execution commands
    # Add more as needed (e.g., 'javascript': ['node', '-e'])
    commands = {
        "python": ["python3", "-c"],
        "bash": ["bash", "-c"]
    }

    # raise HTTPException(status_code=438, detail=f"Random test error for debugging purposes.")

    try:
        validate_code_integrity(request.language.lower(), request.code)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid code grammar: {str(e)}")

    try:
        imports = "from weather import get_forecast, get_alerts\n"
        full_code = imports + request.code
        script_dir = Path(__file__).resolve().parent

        process = await asyncio.create_subprocess_exec(
            *commands[request.language.lower()],
            full_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_dir
        )

        try:
            # Wait for completion with a timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            process.kill()
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Execution timed out.")

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logging.error(f"Execution error:\n{error_msg}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Execution error: {error_msg}")
        
        response.status_code = status.HTTP_200_OK
        return {
            "stdout": stdout.decode()
        }
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Error: {str(e)}")

def validate_code_integrity(language: str, code: str):
    # 1. Language Label Verification (Heuristics)
    markers = {
        "python": [r"def\s+\w+\(", r"import\s+\w+", r"print\(", r"self\."],
        "bash": [r"echo\s+", r"\$\(", r"if\s+\[", r"sudo\s+", r"apt-get"]
    }
    
    if language in markers:
        # Check if at least one marker exists for the labeled language
        if not any(re.search(pattern, code) for pattern in markers[language]):
            raise ValueError(f"Code content does not appear to match the '{language}' label.")

    # 2. Syntax Validation (Dry Run)
    if language == "python":
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Python Syntax Error: {e.msg} at line {e.lineno}")
            
    elif language == "bash":
        check = subprocess.run(["bash", "-n", "-c", code], capture_output=True, text=True)
        if check.returncode != 0:
            raise ValueError(f"Bash Syntax Error: {check.stderr.strip()}")

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
