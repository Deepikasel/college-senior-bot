from dataclasses import dataclass
from typing import List, Tuple
from .retriever import TfidfRetriever
from .data_loader import load_qa
import random

SENIOR_PREFIXES = [
    "Listen junior:",
    "Friendly warning:",
    "Senior insight:",
    "Shortcut to survival:",
    "From painful experience:"
]

SENIOR_SUFFIXES = [
    "You’ll thank me later.",
    "Yes, really.",
    "Trust me, this saves semesters.",
    "Classic mistake—skip it.",
    "Consider this free coaching."
]

SAFE_GUARDRAILS = [
    "I can’t help with anything harmful, unethical, or academic misconduct.",
    "Keep it respectful and within your college’s rules.",
]

@dataclass
class SeniorBot:
    retriever: TfidfRetriever
    answers: List[str]
    spice: int = 2  # 0 = mild, 1 = normal, 2 = spicy

    @staticmethod
    def build(data_path: str, model_dir: str = "models"):
        # train retriever on questions
        qs, ans = load_qa(data_path)
        ret = TfidfRetriever().fit(qs)
        ret.save(model_dir)
        return SeniorBot(ret, ans)

    @staticmethod
    def load(model_dir: str = "models", data_path: str = "data/college_qa.jsonl"):
        # load trained retriever + answers
        bot = SeniorBot(TfidfRetriever().load(model_dir), load_qa(data_path)[1])
        return bot

    def _persona_wrap(self, core: str) -> str:
        spice = max(0, min(2, self.spice))
        prefix = random.choice(SENIOR_PREFIXES) if spice >= 1 else "Tip:"
        suffix = (random.choice(SENIOR_SUFFIXES) if spice == 2 else "")
        return f"{prefix} {core} {suffix}".strip()

    def reply(self, user_text: str) -> Tuple[str, List[Tuple[str, float]]]:
        # safety guardrails
        lowered = user_text.lower()
        if any(x in lowered for x in ["cheat", "leak paper", "hack grade", "proxy attendance"]):
            msg = "Nope. " + random.choice(SAFE_GUARDRAILS)
            return self._persona_wrap(msg), []

        # retrieve best answers
        hits = self.retriever.top_k(user_text, k=3)
        best_idx, best_sim = hits[0]
        core = self.answers[best_idx]

        # if similarity too low, use a generic witty fallback
        if best_sim < 0.12:
            core = "Start small and start now. Future-you will send a thank-you email."

        expl = [(self.retriever.docs[i], s) for i, s in hits]
        return self._persona_wrap(core), expl
