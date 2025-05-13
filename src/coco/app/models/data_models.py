from pydantic import BaseModel
from typing import List

class DataInput(BaseModel):
    text: List[str]

class StatusOutput(BaseModel):
    calculating: bool
    data_count: int
