from fastapi import APIRouter, HTTPException
from coco.app.services.lotw_service import LoTWService
from coco.app.services.kafka_producer import publish_trust_scores

router = APIRouter(prefix="/lotw", tags=["LoTw"])
service = LoTWService()

@router.post("/calculate")
def calculate_lotw():
    """
    Calculates LoTW scores for the data currently stored in the application.

    LoTW scores are calculated for all labels in the stored data. The scores are then
    published to a Kafka topic for further processing.

    Returns a JSON object with the following keys:
        - message: A message indicating the result of the calculation
        - nLoTw: A dictionary containing the nLoTW values for each label
        - cLoTw: A dictionary containing the cLoTW values for each label

    Raises a 404 error if the stored data is empty.
    """
    try:
        nlotw, clotw = service.compute_clotw_scores()
        publish_trust_scores(nlotw, clotw)
        return {"message": "nLoTw calculated and published", "nLoTw": nlotw, "cLoTw": clotw}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))