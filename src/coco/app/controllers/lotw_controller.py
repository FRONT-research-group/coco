from fastapi import APIRouter, HTTPException
from coco.app.services.lotw_service import LoTWService
from coco.app.services.kafka_producer import publish_trust_scores

router = APIRouter(prefix="/lotw", tags=["LoTw"])
service = LoTWService()

@router.post("/calculate")
def calculate_lotw():
    try:
        nlotw, clotw = service.compute_clotw_scores()
        publish_trust_scores(nlotw, clotw)
        return {"message": "nLoTw calculated and published", "nLoTw": nlotw, "cLoTw": clotw}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))