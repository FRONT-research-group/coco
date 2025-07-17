import pytest
from unittest.mock import patch
from coco.app.services.lotw_service import LoTWService
from coco.app.models.lotw_models import LabeledText
from coco.app.core import state

@pytest.fixture
def sample_data():
    return [
        LabeledText(label="Privacy", text="Respect user data"),
        LabeledText(label="Reliability", text="Minimize service outages"),
        LabeledText(label="Security", text="Encrypt all data"),
        LabeledText(label="Privacy", text="Use anonymization"),
    ]

@pytest.fixture(autouse=True)
def clear_state():
    
    state.stored_data.clear()
    state.nlotw_score = None
    state.calculating = False

@patch("coco.app.services.lotw_service.LoTWService.predict_score", return_value=50.0)
def test_calculate_nlotw(mock_predict, sample_data):
    service = LoTWService()
    nlotw = service.calculate_nlotw(sample_data)
    assert isinstance(nlotw, dict)
    assert all(isinstance(score, float) for score in nlotw.values())
    assert abs(sum(nlotw.values()) - 100.0) < 1e-6

def test_compute_wtf():
    service = LoTWService()
    scores_per_label = {
        "Privacy": [1, 2],
        "Security": [1],
    }
    wtf = service.compute_wtf(scores_per_label)
    assert isinstance(wtf, dict)
    assert abs(sum(wtf.values()) - 1.0) < 1e-6
