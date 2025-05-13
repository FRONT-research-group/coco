from fastapi import APIRouter
from coco.app.models.data_models import DataInput, StatusOutput

from coco.app.services import service as clotw_service

router = APIRouter(prefix="/data", tags=["Data"])

@router.post("/submit")
def submit_data(data: DataInput):
    count = clotw_service.submit_data(data.text)
    return {"message": "Data received successfully", "count": count}

@router.get("/status", response_model=StatusOutput)
def get_status():
    calculating, count = clotw_service.get_status()
    return StatusOutput(calculating=calculating, data_count=count)
