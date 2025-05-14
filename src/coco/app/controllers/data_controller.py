from fastapi import APIRouter
from coco.app.models.lotw_models import DataInput, StatusOutput
from coco.app.core import state

router = APIRouter(prefix="/data", tags=["Data"])

@router.post("/submit")
def submit_data(data: DataInput):
    state.stored_data.extend(data.data)
    state.calculating = True
    return {"message": "Data received successfully", "count": len(state.stored_data)}

@router.get("/status", response_model=StatusOutput)
def get_status():
    return StatusOutput(calculating=state.calculating, data_count=len(state.stored_data))
