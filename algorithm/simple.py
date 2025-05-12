import json


def analyze(str_data: str):
    data = json.loads(str_data)
    current = data["current_vector"][1]
    target = data["target_vector"][1]
    engine = 0

    if abs(current - target) <= 1:  # остаемся на той же высоте
        engine = 75
    elif current < target:          # летим вверх
        engine = 100
    else:                           # летим вниз
        engine = 0

    print(f"Current = {current}, target = {target}, engine = {engine}")

    result = {"id": data["id"], "engines": {
        "fr": engine,
        "fl": engine,
        "br": engine,
        "bl": engine,
        "rf": engine,
        "rb": engine,
        "lf": engine,
        "lb": engine,
    }}

    return json.dumps(result)
