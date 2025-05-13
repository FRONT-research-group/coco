from fastapi import APIRouter, HTTPException
from coco.app.models.lotw_models import ClotwOutput, NlotwOutput

from coco.app.services import service as clotw_service

router = APIRouter(prefix="/lotw", tags=["LoTw"])

@router.post("/calculate")
def calculate_lotw():
    try:
        clotw, _ = clotw_service.compute_clotw_scores()
        return {"message": "cLoTw calculated successfully", "cLoTw": clotw}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/clotw", response_model=ClotwOutput)
def get_clotw():
    try:
        clotw = clotw_service.get_clotw()
        return ClotwOutput(cLoTw=clotw)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/nlotw", response_model=NlotwOutput)
def get_nlotw():
    try:
        nlotw = clotw_service.get_nlotw()
        return NlotwOutput(nLoTw=nlotw)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
