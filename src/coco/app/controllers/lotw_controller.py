from fastapi import APIRouter, HTTPException
from coco.app.models.lotw_models import NlotwOutput
from coco.app.core import state
from coco.app.services.lotw_service import LoTWService

router = APIRouter(prefix="/lotw", tags=["LoTw"])
service = LoTWService()

@router.post("/calculate")
def calculate_lotw():
    try:
        nlotw = service.compute_clotw_scores()
        return {"message": "cLoTw calculated successfully", "nLoTw": nlotw}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/nlotw", response_model=NlotwOutput)
def get_nlotw():
    if state.nlotw_score is None:
        raise HTTPException(status_code=404, detail="nLoTw not yet calculated")
    return NlotwOutput(nLoTw=state.nlotw_score)
