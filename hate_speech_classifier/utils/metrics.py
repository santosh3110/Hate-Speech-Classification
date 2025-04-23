import os

def save_best_accuracy(path: str, accuracy: float):
    with open(path, "w") as f:
        f.write(str(accuracy))

def load_best_accuracy(path: str) -> float:
    if os.path.exists(path):
        with open(path, "r") as f:
            return float(f.read().strip())
    return None
