"""
localization_ranker.py
======================
Leave-One-Out (LOO) GBM-based transaction localization ranker.

This module implements the content-aware fusion pipeline for transaction-level
localization as described in Section IV.D of the paper. It combines:

    1. Neural attention scores from UnifiedTMIL.
    2. Engineered on-chain features (counterparty reputation, degree, amount
       spike, novelty, recency, etc.).
    3. A Gradient Boosting Classifier trained in a Leave-One-Out (LOO) fashion
       to avoid data leakage from counterparty reputation features.

The LOO protocol ensures that the reputation feature for bag *i* is computed
using all bags **except** bag *i*, preventing the model from observing its own
label signal during inference.

Usage::

    python scripts/extract_attn_and_localize.py

Outputs:
    results/loc_fusion_marginal.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------------------------------------------------------
# Path setup — allow running as a standalone script
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# ---------------------------------------------------------------------------
# Patch __main__.Vocab so that pickled vocab.pkl deserializes correctly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "src"))
import vocab_def  # noqa: E402
sys.modules["__main__"].Vocab = vocab_def.Vocab  # type: ignore[attr-defined]


def load_pkl(path: str) -> Any:
    """Deserialize a pickle file.

    Args:
        path (str): Absolute path to the ``.pkl`` file.

    Returns:
        Any: The deserialized Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def usable(bag: dict) -> tuple[int, list[int]]:
    """Return the number of candidate transactions and ground-truth indices.

    A bag is *usable* for localization evaluation if it has at least one
    interior ground-truth transaction (i.e., the GT is not the last position).

    Args:
        bag (dict): A bag dictionary with keys ``"length"``, ``"gt_indices"``.

    Returns:
        tuple[int, list[int]]: ``(n_candidates, gt_indices)`` where
            ``n_candidates`` is the number of valid candidate transactions and
            ``gt_indices`` is the list of ground-truth positions within
            ``[0, n_candidates)``.
    """
    nc = bag.get("length", 0)
    gt = [g for g in bag.get("gt_indices", []) if g < nc - 1]
    return nc, gt


def build_base_features(
    bag: dict,
    n_candidates: int,
    attention: np.ndarray,
    use_attention: bool = True,
) -> np.ndarray:
    """Construct the per-transaction feature matrix for a single bag.

    Features (in order):
        - log-amount (signed)
        - attention-weighted context projection (scalar)
        - attention rank (normalized)
        - novelty (1 if first occurrence of counterparty)
        - inbound flag
        - pre-last-position flag
        - attention weight (if ``use_attention``)
        - attention rank (if ``use_attention``)
        - zero-outbound flag
        - outbound flag
        - average value of counterparty
        - running maximum amount
        - log-amount (unsigned)
        - log-delta-t
        - inbound value
        - distance to last transaction
        - counterparty frequency

    Args:
        bag (dict): Bag dictionary.
        n_candidates (int): Number of valid candidate transactions.
        attention (np.ndarray): Per-transaction attention weights from UnifiedTMIL.
        use_attention (bool): Whether to include attention-derived features.

    Returns:
        np.ndarray: Feature matrix of shape ``(n_candidates, n_features)``.
    """
    ids = bag["input_ids"][:n_candidates]
    io_flags = bag["input_io"][:n_candidates]
    amounts = np.array(bag["input_amounts"][:n_candidates], dtype=float)
    delta_ts = np.array(bag["delta_ts"][:n_candidates], dtype=float)
    n = n_candidates

    la = np.log1p(np.abs(amounts)) * np.sign(amounts)
    z = np.zeros(n)
    rank = np.arange(n, dtype=float) / max(n - 1, 1)
    nov = np.zeros(n)
    seen: set = set()
    for k, c in enumerate(ids):
        if c not in seen:
            nov[k] = 1.0
            seen.add(c)
    inb = (np.array(io_flags) == 2).astype(float)
    prel = np.zeros(n)
    if n > 1:
        prel[-2] = 1.0

    zero_out = (np.array(io_flags) == 0).astype(float)
    outb = (np.array(io_flags) == 1).astype(float)
    avc = np.zeros(n)
    cnt: dict = {}
    sm: dict = {}
    for k, (c, a) in enumerate(zip(ids, amounts)):
        cnt[c] = cnt.get(c, 0) + 1
        sm[c] = sm.get(c, 0.0) + a
        avc[k] = sm[c] / cnt[c]
    runmax = np.maximum.accumulate(np.abs(amounts))
    lz = np.log1p(np.abs(amounts))
    ldt = np.log1p(delta_ts)
    invalue = np.abs(amounts) * inb
    dist_nl = np.arange(n, 0, -1, dtype=float) / n
    cc = Counter(ids)
    cpf = np.array([cc[c] for c in ids], float) / n

    cols = [la, z, rank, nov, inb, prel]
    if use_attention:
        at = attention[:n_candidates] if len(attention) >= n_candidates else np.zeros(n_candidates)
        ar = np.argsort(np.argsort(at)) / max(n - 1, 1)
        cols += [at, ar]
    cols += [zero_out, outb, avc, runmax, lz, ldt, invalue, dist_nl, cpf]

    return np.column_stack(cols)


def best_rank(scores: np.ndarray, gt_indices: list[int]) -> int:
    """Return the best (lowest) rank achieved by any ground-truth transaction.

    Args:
        scores (np.ndarray): Per-transaction scores (higher = more suspicious).
        gt_indices (list[int]): Ground-truth transaction indices.

    Returns:
        int: 0-based rank of the best-ranked GT transaction.
    """
    order = np.argsort(-np.asarray(scores))
    rank_map = {pos: r for r, pos in enumerate(order)}
    return min(rank_map[g] for g in gt_indices)


def loo_reputation(
    bags: list[dict],
    bag_ids: list[np.ndarray],
    alpha: float = 5.0,
) -> list[np.ndarray]:
    """Compute Leave-One-Out counterparty reputation for all bags.

    For each bag *i*, the reputation of counterparty *c* is:

        r(c, i) = (phishing_count(c) - 1[y_i=1]) / (total_count(c) - 1)

    where the bag's own contribution is subtracted (sum-minus-i trick).

    Args:
        bags (list[dict]): List of bag dictionaries.
        bag_ids (list[np.ndarray]): Pre-extracted counterparty ID arrays.
        alpha (float): Laplace smoothing strength. Default: 5.0.

    Returns:
        list[np.ndarray]: Per-bag counterparty reputation arrays.
    """
    g: dict = defaultdict(float)
    t: dict = defaultdict(float)
    G = T = 0.0

    for bag, ids in zip(bags, bag_ids):
        for c in ids:
            t[c] += 1
            g[c] += bag["label"]
        T += len(ids)
        G += bag["label"] * len(ids)

    prior = G / max(T, 1)
    out = []
    for bag, ids in zip(bags, bag_ids):
        gi: dict = defaultdict(float)
        ti: dict = defaultdict(float)
        Gi = G
        Ti = T
        for c in ids:
            gi[c] += bag["label"]
            ti[c] += 1
            Gi -= bag["label"]
            Ti -= 1
        p = Gi / max(Ti, 1)
        rep = np.array(
            [(g[c] - gi[c] + alpha * p) / ((t[c] - ti[c]) + alpha) for c in ids]
        )
        out.append(rep)
    return out
