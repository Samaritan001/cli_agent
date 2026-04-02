from pydantic import BaseModel
from typing import Optional

class CodeRequest(BaseModel):
    language: str
    code: str


class CommandRequest(BaseModel):
    command: str
    id: str
    fetch_manual: Optional[bool] = True
    server_name: Optional[str] = None
    language: Optional[str] = None
    code: Optional[str] = None
