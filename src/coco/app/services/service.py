from coco.app.core import state

# Placeholder functions for actual logic
def calculate_clotw(data: list[str]) -> float:
    return sum(len(x) for x in data) / max(len(data), 1)

def calculate_nlotw(data: list[str]) -> float:
    return sum(x.count(' ') + 1 for x in data) / max(len(data), 1)

def submit_data(texts: list[str]) -> int:
    state.stored_data.extend(texts)
    state.calculating = True
    return len(state.stored_data)

def compute_clotw_scores() -> tuple[float, float]:
    if not state.stored_data:
        raise ValueError("No data available")

    state.nlotw_score = calculate_nlotw(state.stored_data)
    state.clotw_score = calculate_clotw(state.stored_data)
    state.calculating = False
    return state.clotw_score, state.nlotw_score

def get_clotw() -> float:
    if state.clotw_score is None:
        raise ValueError("cLoTw not calculated yet")
    return state.clotw_score

def get_nlotw() -> float:
    if state.nlotw_score is None:
        raise ValueError("nLoTw not calculated yet")
    return state.nlotw_score

def get_status() -> tuple[bool, int]:
    return state.calculating, len(state.stored_data)
