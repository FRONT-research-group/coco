from pydantic import BaseModel
from typing import List, Dict

from coco.app.models.data_models import TrustFunction

class LabeledText(BaseModel):
    label: TrustFunction
    text: str

class DataInput(BaseModel):
    data: List[LabeledText]

class StatusOutput(BaseModel):
    calculating: bool
    data_count: int

class NlotwOutput(BaseModel):
    nLoTw: Dict[str, float]
