"""Deterministic one-at-a-time sensitivity and optional seeded weight scenarios."""
import numpy as np
import pandas as pd

from opportunities import COMPONENTS, rank


def sensitivity(table, weights, spans, scenarios=300, seed=2026):
    weights = np.asarray(weights, float)
    nominal = rank(table, weights).Score
    draws = np.random.default_rng(seed).uniform(.8, 1.2, (scenarios, len(weights)))*weights
    draws = 100*draws/draws.sum(axis=1, keepdims=True)
    scores = pd.DataFrame([rank(table, draw).Score for draw in draws])
    ranks = scores.rank(axis=1, method="min", ascending=False)
    report = pd.DataFrame({"Scenario_score_min": scores.min(),
                           "Scenario_score_max": scores.max(),
                           "Best_rank": ranks.min(), "Worst_rank": ranks.max(),
                           "First_place_fraction": ranks.eq(1).mean().where(nominal.notna())})
    for field in COMPONENTS:
        if field not in spans:
            continue
        lo, hi = table.copy(), table.copy()
        lo[field] = (table[field]-spans[field]/2).clip(0, 1)
        hi[field] = (table[field]+spans[field]/2).clip(0, 1)
        report[f"{field}_score_span"] = rank(hi, weights).Score-rank(lo, weights).Score
    return report

