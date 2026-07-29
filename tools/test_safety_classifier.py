#!/usr/bin/env python3
"""
10 hand-crafted tests for EvidenceLinkedSafetyClassifier with known expected verdicts.

Run directly:  python tools/test_safety_classifier.py
Or via pytest: pytest tools/test_safety_classifier.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from safety_classifier import (
    EvidenceLinkedSafetyClassifier,
    FeatureLabel,
    SafetyClass,
)

# ── Shared label store ────────────────────────────────────────────────────────
#
# In production these come from AutoInterp or human review of max-activating
# examples (see data/sae-analysis/find_max_activating.py).  Here we define a
# minimal hand-crafted set sufficient for deterministic testing.

LABELS: dict[int, FeatureLabel] = {
    # Safe features (do not affect safety verdict)
    0: FeatureLabel(0, "neutral punctuation",          SafetyClass.SAFE,      weight=1.0),
    1: FeatureLabel(1, "common grammatical structure", SafetyClass.SAFE,      weight=1.0),
    2: FeatureLabel(2, "benign code syntax",           SafetyClass.SAFE,      weight=1.0),
    # Uncertain features (flag for human review)
    10: FeatureLabel(10, "sensitive personal information",   SafetyClass.UNCERTAIN, weight=1.0),
    11: FeatureLabel(11, "potentially deceptive framing",    SafetyClass.UNCERTAIN, weight=1.5),
    12: FeatureLabel(12, "dual-use technical content",       SafetyClass.UNCERTAIN, weight=1.0),
    # Flagged features (block / escalate)
    20: FeatureLabel(20, "explicit harmful instructions", SafetyClass.FLAGGED, weight=2.0),
    21: FeatureLabel(21, "hate speech patterns",          SafetyClass.FLAGGED, weight=2.0),
    22: FeatureLabel(22, "violence promotion",            SafetyClass.FLAGGED, weight=1.5),
    23: FeatureLabel(23, "self-harm content",             SafetyClass.FLAGGED, weight=2.0),
}

clf = EvidenceLinkedSafetyClassifier(LABELS, activation_threshold=0.0)


# ── Test 1 ───────────────────────────────────────────────────────────────────

def test_empty_activations():
    """No features fire at all → safe with maximum confidence."""
    result = clf.classify({})

    assert result.verdict == SafetyClass.SAFE,      f"expected safe, got {result.verdict}"
    assert result.confidence == 0.95,               f"expected 0.95, got {result.confidence}"
    assert result.n_flagged == 0
    assert result.n_uncertain == 0
    assert result.n_safe == 0
    assert result.active_features == []


# ── Test 2 ───────────────────────────────────────────────────────────────────

def test_only_safe_features():
    """
    Only benign labeled features fire → safe.
    Confidence drops slightly (0.90) because labeled features did fire,
    we just happen to know they're safe.
    """
    result = clf.classify({0: 1.2, 1: 0.8, 2: 2.1})

    assert result.verdict == SafetyClass.SAFE,  f"expected safe, got {result.verdict}"
    assert result.confidence == 0.90,           f"expected 0.90, got {result.confidence}"
    assert result.n_safe == 3
    assert result.n_flagged == 0
    assert result.n_uncertain == 0


# ── Test 3 ───────────────────────────────────────────────────────────────────

def test_single_flagged_feature_strong():
    """
    One strong flagged feature alongside a safe feature → flagged.
    Confidence ≥ 0.60 (single flagged: 0.60 + 0.10×1 = 0.70).
    """
    result = clf.classify({0: 1.0, 20: 3.5})

    assert result.verdict == SafetyClass.FLAGGED,  f"expected flagged, got {result.verdict}"
    assert result.n_flagged == 1
    assert result.n_uncertain == 0
    assert result.confidence >= 0.60,              f"confidence too low: {result.confidence}"
    # The flagged feature should be first (highest activation×weight)
    assert result.active_features[0].feature_id == 20


# ── Test 4 ───────────────────────────────────────────────────────────────────

def test_flagged_at_threshold_not_active():
    """
    Flagged feature activation equals the threshold (0.0) — not strictly above,
    so it is ignored.  Only a safe feature fires → safe verdict.
    """
    result = clf.classify({20: 0.0, 0: 1.5})

    assert result.verdict == SafetyClass.SAFE,  f"expected safe, got {result.verdict}"
    assert result.n_flagged == 0
    assert result.n_safe == 1


# ── Test 5 ───────────────────────────────────────────────────────────────────

def test_single_uncertain_feature():
    """One uncertain feature, no flagged → uncertain."""
    result = clf.classify({10: 1.2})

    assert result.verdict == SafetyClass.UNCERTAIN, f"expected uncertain, got {result.verdict}"
    assert result.n_uncertain == 1
    assert result.n_flagged == 0
    # confidence formula: 0.50 + 0.10 × 1 = 0.60
    assert result.confidence == 0.60, f"expected 0.60, got {result.confidence}"


# ── Test 6 ───────────────────────────────────────────────────────────────────

def test_safe_and_uncertain_features():
    """
    Safe features cannot override uncertain ones.
    Any uncertain feature firing lifts the verdict to uncertain.
    """
    result = clf.classify({0: 2.0, 1: 1.5, 11: 0.8})

    assert result.verdict == SafetyClass.UNCERTAIN, f"expected uncertain, got {result.verdict}"
    assert result.n_safe == 2
    assert result.n_uncertain == 1
    assert result.n_flagged == 0


# ── Test 7 ───────────────────────────────────────────────────────────────────

def test_uncertain_and_flagged_flagged_wins():
    """
    Both uncertain and flagged features fire → flagged (flagged takes precedence).
    Confidence is reduced by the mixed-evidence penalty because the uncertain
    weighted score (4.5) dominates the flagged weighted score (0.75).
    """
    result = clf.classify({11: 2.0, 12: 1.5, 22: 0.5})

    assert result.verdict == SafetyClass.FLAGGED,  f"expected flagged, got {result.verdict}"
    assert result.n_uncertain == 2
    assert result.n_flagged == 1
    # Confidence should reflect mixed signal (lower than a clean flagged result)
    assert result.confidence < 0.70, f"expected mixed-penalty reduction, got {result.confidence}"


# ── Test 8 ───────────────────────────────────────────────────────────────────

def test_multiple_flagged_features_high_confidence():
    """
    Three flagged features firing with no uncertain features → flagged,
    high confidence ≥ 0.85 (formula: 0.60 + 0.10×3 = 0.90).
    """
    result = clf.classify({20: 2.5, 21: 3.0, 22: 1.8})

    assert result.verdict == SafetyClass.FLAGGED, f"expected flagged, got {result.verdict}"
    assert result.n_flagged == 3
    assert result.n_uncertain == 0
    assert result.confidence > 0.80, f"expected high confidence, got {result.confidence}"


# ── Test 9 ───────────────────────────────────────────────────────────────────

def test_many_safe_plus_single_uncertain():
    """
    Even when safe features dominate by count, a single uncertain feature
    shifts the verdict to uncertain.
    """
    result = clf.classify({0: 3.0, 1: 2.5, 2: 1.8, 10: 0.6})

    assert result.verdict == SafetyClass.UNCERTAIN, f"expected uncertain, got {result.verdict}"
    assert result.n_safe == 3
    assert result.n_uncertain == 1
    assert result.n_flagged == 0


# ── Test 10 ──────────────────────────────────────────────────────────────────

def test_dense_numpy_array_multiple_flagged():
    """
    Dense numpy array input: three flagged features fire at high activation,
    one safe feature also fires.  Verdict: flagged with confidence ≥ 0.85.
    """
    acts = np.zeros(100)
    acts[0]  = 1.0   # safe feature 0
    acts[20] = 4.2   # flagged: explicit harmful instructions  (weight 2.0)
    acts[21] = 2.1   # flagged: hate speech patterns           (weight 2.0)
    acts[23] = 1.5   # flagged: self-harm content              (weight 2.0)

    result = clf.classify(acts)

    assert result.verdict == SafetyClass.FLAGGED, f"expected flagged, got {result.verdict}"
    assert result.n_flagged == 3
    assert result.n_uncertain == 0
    assert result.confidence >= 0.85, f"expected ≥0.85, got {result.confidence}"
    # as_dict should be JSON-serialisable (no numpy types)
    d = result.as_dict()
    assert d["verdict"] == "flagged"
    assert isinstance(d["confidence"], float)
    assert all(isinstance(f["activation"], float) for f in d["active_features"])


# ── Runner ───────────────────────────────────────────────────────────────────

TESTS = [
    test_empty_activations,
    test_only_safe_features,
    test_single_flagged_feature_strong,
    test_flagged_at_threshold_not_active,
    test_single_uncertain_feature,
    test_safe_and_uncertain_features,
    test_uncertain_and_flagged_flagged_wins,
    test_multiple_flagged_features_high_confidence,
    test_many_safe_plus_single_uncertain,
    test_dense_numpy_array_multiple_flagged,
]


def _run_all() -> int:
    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    total = passed + failed
    print(f"\n{passed}/{total} passed", "✓" if failed == 0 else "✗")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
