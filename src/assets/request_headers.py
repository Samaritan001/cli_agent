from pydantic import BaseModel
from typing import Optional

class CodeRequest(BaseModel):
    language: str
    code: str


class CommandRequest(BaseModel):
    command: str
    server_name: Optional[str] = None
    server_summary: Optional[str] = None
    language: Optional[str] = None
    code: Optional[str] = None
