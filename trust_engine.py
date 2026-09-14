"""
TrustEngine — combines multiple real signals into ONE Trust Score per place
(hotel / restaurant / transport listing).

Matches the architecture diagram: Rating + Reviews(volume) + Freshness +
Credibility + Sentiment + Safety + Consistency -> one blended score.

WHAT THIS REUSES (nothing invented from scratch):
  - Credibility factor = output of the fake-review model from
    train_fake_review_model.py (fake_review_model.pkl / _vectorizer.pkl)
  - Sentiment factor = VADER (vaderSentiment package) — a real, pretrained
    sentiment lexicon tuned for short informal text like reviews; no
    training needed, `pip install vaderSentiment`
  - Safety factor = a HOOK for your project's existing risk-classifier
    .pkl (see model/*.pkl in your uploaded project) — left as a pluggable
    function rather than guessed, since I don't know that model's exact
    input feature schema. Wire it in once you confirm the schema (see
    safety_score_fn below).

HONEST SCOPE: this produces ONE score per place, the same for every user —
it is a Trust Score / ranking signal, NOT personalized recommendation.
Personalization needs per-user history, which is a separate, harder
problem (see our discussion on the "cold start" issue).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
import math

import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sentiment_analyzer = SentimentIntensityAnalyzer()


@dataclass
class Review:
    text: str
    rating: float          # 1-5 stars
    source: str             # e.g. "google" or "app" — used for the consistency factor
    posted_at: datetime      # timezone-aware


@dataclass
class TrustScoreBreakdown:
    trust_score: float                  # 0-100 — the final blended score
    rating_score: float                 # each sub-score also 0-100, for showing your work
    volume_score: float
    freshness_score: float
    credibility_score: float
    sentiment_score: float
    consistency_score: float
    safety_score: Optional[float]       # None if no safety hook was wired in
    genuine_review_count: int
    total_review_count: int


class TrustEngine:
    def __init__(
        self,
        fake_review_model_path: str = "fake_review_model.pkl",
        fake_review_vectorizer_path: str = "fake_review_vectorizer.pkl",
        safety_score_fn: Optional[Callable[[list], float]] = None,
        weights: Optional[dict] = None,
    ):
        self._model = joblib.load(fake_review_model_path)
        self._vectorizer = joblib.load(fake_review_vectorizer_path)

        # Optional: pass a function(genuine_reviews) -> float in [0, 1] that
        # calls your EXISTING risk-classifier .pkl for this place's
        # area/location. Left as a hook, not guessed — wire it in once you
        # confirm that model's real input schema.
        self._safety_score_fn = safety_score_fn

        # Tune these with your team — they must sum to 1.0 across whichever
        # factors are active. If no safety hook is wired, its weight is
        # redistributed proportionally (see compute() below) rather than
        # silently shrinking the total score.
        self.weights = weights or {
            "rating": 0.25,
            "volume": 0.10,
            "freshness": 0.15,
            "credibility": 0.20,
            "sentiment": 0.10,
            "consistency": 0.10,
            "safety": 0.10,
        }

    # ---------------- individual factors ----------------

    def _classify_genuine(self, reviews: list) -> list:
        if not reviews:
            return []
        texts = [r.text for r in reviews]
        vecs = self._vectorizer.transform(texts)
        preds = self._model.predict(vecs)  # 1 = fake, 0 = genuine (per training script's label convention)
        return [r for r, p in zip(reviews, preds) if p == 0]

    def _rating_score(self, genuine: list) -> float:
        if not genuine:
            return 0.0
        avg = sum(r.rating for r in genuine) / len(genuine)
        return avg / 5.0

    def _volume_score(self, genuine: list) -> float:
        # Log-scaled so 5 vs 5000 reviews isn't a linear, easily-gamed
        # difference — caps out around 200 genuine reviews.
        n = len(genuine)
        return min(math.log10(n + 1) / math.log10(200), 1.0)

    def _freshness_score(self, genuine: list) -> float:
        if not genuine:
            return 0.0
        now = datetime.now(timezone.utc)
        half_life_days = 180  # a review "counts half as much" after ~6 months — tune with your team
        decays = []
        for r in genuine:
            age_days = max((now - r.posted_at).days, 0)
            decays.append(0.5 ** (age_days / half_life_days))
        return sum(decays) / len(decays)

    def _credibility_score(self, genuine_count: int, total_count: int) -> float:
        if total_count == 0:
            return 0.0
        return genuine_count / total_count

    def _sentiment_score(self, genuine: list) -> float:
        if not genuine:
            return 0.0
        compounds = [_sentiment_analyzer.polarity_scores(r.text)["compound"] for r in genuine]
        avg_compound = sum(compounds) / len(compounds)  # VADER compound ranges -1 to 1
        return (avg_compound + 1) / 2  # normalize to 0-1

    def _consistency_score(self, genuine: list) -> float:
        # Agreement between sources (e.g. Google rating vs in-app rating) —
        # low agreement is itself a useful signal, not just noise.
        by_source: dict = {}
        for r in genuine:
            by_source.setdefault(r.source, []).append(r.rating)
        if len(by_source) < 2:
            return 1.0  # only one source present — nothing to compare, don't penalize
        averages = [sum(v) / len(v) for v in by_source.values()]
        spread = max(averages) - min(averages)  # 0 (full agreement) to 4 (max on a 1-5 scale)
        return max(1.0 - (spread / 4.0), 0.0)

    # ---------------- put it all together ----------------

    def compute(self, reviews: list) -> TrustScoreBreakdown:
        genuine = self._classify_genuine(reviews)

        components = {
            "rating": self._rating_score(genuine),
            "volume": self._volume_score(genuine),
            "freshness": self._freshness_score(genuine),
            "credibility": self._credibility_score(len(genuine), len(reviews)),
            "sentiment": self._sentiment_score(genuine),
            "consistency": self._consistency_score(genuine),
        }

        active_weights = dict(self.weights)
        safety = None
        if self._safety_score_fn is not None:
            safety = self._safety_score_fn(genuine)
            components["safety"] = safety
        else:
            # No safety hook wired yet — redistribute its weight
            # proportionally across the active factors instead of silently
            # under-weighting the total (which would make every score look
            # artificially low relative to the intended 0-100 scale).
            safety_weight = active_weights.pop("safety", 0.0)
            remaining_total = sum(active_weights.values())
            if remaining_total > 0:
                for k in active_weights:
                    active_weights[k] += safety_weight * (active_weights[k] / remaining_total)

        blended = sum(components[k] * active_weights[k] for k in components)

        return TrustScoreBreakdown(
            trust_score=round(blended * 100, 1),
            rating_score=round(components["rating"] * 100, 1),
            volume_score=round(components["volume"] * 100, 1),
            freshness_score=round(components["freshness"] * 100, 1),
            credibility_score=round(components["credibility"] * 100, 1),
            sentiment_score=round(components["sentiment"] * 100, 1),
            consistency_score=round(components["consistency"] * 100, 1),
            safety_score=round(safety * 100, 1) if safety is not None else None,
            genuine_review_count=len(genuine),
            total_review_count=len(reviews),
        )


# ---------------- example usage ----------------
if __name__ == "__main__":
    from datetime import timedelta

    engine = TrustEngine()  # loads fake_review_model.pkl + _vectorizer.pkl from the current folder

    sample_reviews = [
        Review(
            text="Great location, friendly staff, room was clean and spacious. Would stay again.",
            rating=5,
            source="google",
            posted_at=datetime.now(timezone.utc) - timedelta(days=10),
        ),
        Review(
            text="This hotel is the best hotel ever amazing perfect wonderful staff perfect location",
            rating=5,
            source="app",
            posted_at=datetime.now(timezone.utc) - timedelta(days=400),
        ),
    ]

    result = engine.compute(sample_reviews)
    print(result)
