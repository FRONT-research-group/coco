from enum import Enum
from pydantic import BaseModel
from typing import List

class DataInput(BaseModel):
    text: List[str]

class StatusOutput(BaseModel):
    calculating: bool
    data_count: int

class TrustFunction(str, Enum):
    RELIABILITY = "Reliability"
    PRIVACY = "Privacy"
    SECURITY = "Security"
    RESILIENCE = "Resilience"
    SAFETY = "Safety"
