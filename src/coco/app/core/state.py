from typing import List, Dict, Optional
from coco.app.models.lotw_models import LabeledText

stored_data: List[LabeledText] = []
calculating: bool = False
clotw_score: Optional[Dict[str, List[float]]] = None
nlotw_score: Optional[Dict[str, float]] = None
