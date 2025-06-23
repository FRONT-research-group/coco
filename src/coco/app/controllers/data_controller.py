from fastapi import APIRouter
from coco.app.models.lotw_models import DataInput, StatusOutput
from coco.app.core import state

router = APIRouter(prefix="/data", tags=["Data"])

@router.post("/submit")
def submit_data(data: DataInput):
    """
    Submits a batch of labeled data to be processed by the nLoTW and cLoTW algorithms.
    
    Args:
        data (DataInput): The labeled data to be processed, containing a list of LabeledText objects.
    
    Returns:
        A dictionary containing a success message and the total amount of data currently stored.
    """
    state.stored_data.extend(data.data)
    state.calculating = True
    return {"message": "Data received successfully", "count": len(state.stored_data)}

@router.get("/status", response_model=StatusOutput)
def get_status():
    """
    Retrieves the current status of data processing.

    Returns:
        StatusOutput: An object containing the current status of whether data is being calculated
        and the count of data items stored in the system.
    """
    return StatusOutput(calculating=state.calculating, data_count=len(state.stored_data))

@router.get("/reset")
def clear_data():
    """
    Resets the stored data and calculation status to their initial state.
    
    Returns:
        A dictionary containing a success message and the total amount of data currently stored.
    """
    state.stored_data.clear()
    state.calculating = False
    return {"message": "Data cleared successfully", "count": len(state.stored_data)}