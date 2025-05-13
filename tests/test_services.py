import pytest
from coco.app.services import service as clotw_service
from coco.app.core import state

@pytest.fixture(autouse=True)
def clear_state():
    state.stored_data.clear()
    state.clotw_score = None
    state.nlotw_score = None
    state.calculating = False

def test_submit_data():
    count = clotw_service.submit_data(["hello world", "fastapi test"])
    assert count == 2
    assert state.calculating is True

def test_compute_scores():
    clotw_service.submit_data(["a b c", "de fg"])
    clotw, nlotw = clotw_service.compute_clotw_scores()
    assert clotw > 0
    assert nlotw > 0
    assert state.calculating is False

def test_get_status():
    clotw_service.submit_data(["text"])
    calculating, count = clotw_service.get_status()
    assert calculating is True
    assert count == 1
