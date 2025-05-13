from coco.app.services import service as cc

def test_calculate_clotw():
    result = cc.calculate_clotw(["abc", "defg"])
    assert isinstance(result, float)
    assert result > 0

def test_calculate_nlotw():
    result = cc.calculate_nlotw(["one two", "three four five"])
    assert isinstance(result, float)
    assert result > 0
