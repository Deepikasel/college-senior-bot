import json
from pathlib import Path
from typing import List, Dict, Tuple

def load_qa(path: str | Path) -> Tuple[list[str], list[str]]:
    """Load JSONL with fields: q, a. Returns (questions, answers)."""
    path = Path(path)
    questions, answers = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            questions.append(obj["q"].strip())
            answers.append(obj["a"].strip())
    return questions, answers

def to_pairs(path: str | Path) -> List[Dict[str, str]]:
    qs, ans = load_qa(path)
    return [{"q": q, "a": a} for q, a in zip(qs, ans)]
