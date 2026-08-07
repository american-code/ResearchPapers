#!/usr/bin/env python3
"""
Evidence-linked safety classifier using SAE feature activations.

Given a forward pass with SAE activations captured, identifies which labeled
features fire above threshold, then maps active feature labels to a verdict:
safe, uncertain, or flagged.

Returns: verdict, list of active features with labels and activation strengths, confidence.

Verdict precedence (highest wins):
  flagged   — any feature with safety_class=FLAGGED fires above threshold
  uncertain — any feature with safety_class=UNCERTAIN fires; no flagged features fire
  safe      — no safety-relevant (flagged/uncertain) labeled features fire

Confidence reflects signal strength and consistency:
  safe:      0.95 (no labeled features at all), 0.90 (only safe-labeled features)
  uncertain: 0.50 + 0.10 × min(n_uncertain, 4), capped at 0.90
  flagged:   0.60 + 0.10 × min(n_flagged, 4), capped at 0.95,
             then reduced by a mixed-evidence penalty when uncertain weighted
             score exceeds the flagged weighted score
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np


class SafetyClass(str, Enum):
    SAFE = "safe"
    UNCERTAIN = "uncertain"
    FLAGGED = "flagged"


@dataclass
class FeatureLabel:
    """Semantic label and safety classification assigned to one SAE feature."""

    feature_id: int
    label: str           # human-readable description
    safety_class: SafetyClass
    weight: float = 1.0  # importance multiplier used in confidence scoring


@dataclass
class ActiveFeature:
    """A labeled feature that fired above threshold in a given forward pass."""

    feature_id: int
    label: str
    safety_class: SafetyClass
    activation: float
    weight: float


@dataclass
class ClassifierResult:
    """Full output of one classify() call."""

    verdict: SafetyClass
    active_features: list[ActiveFeature]   # sorted by activation×weight descending
    confidence: float                       # 0.0–1.0
    n_flagged: int
    n_uncertain: int
    n_safe: int

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "n_flagged": self.n_flagged,
            "n_uncertain": self.n_uncertain,
            "n_safe": self.n_safe,
            "active_features": [
                {
                    "feature_id": f.feature_id,
                    "label": f.label,
                    "safety_class": f.safety_class.value,
                    "activation": round(f.activation, 4),
                    "weight": f.weight,
                }
                for f in self.active_features
            ],
        }


class EvidenceLinkedSafetyClassifier:
    """
    Maps SAE feature activations to a safety verdict using a labeled feature dictionary.

    Usage:
        labels = {
            42: FeatureLabel(42, "explicit harmful instructions", SafetyClass.FLAGGED, weight=2.0),
            17: FeatureLabel(17, "dual-use chemistry",            SafetyClass.UNCERTAIN),
             3: FeatureLabel( 3, "benign code syntax",            SafetyClass.SAFE),
        }
        clf = EvidenceLinkedSafetyClassifier(labels)
        result = clf.classify(activations)   # dense np.ndarray or sparse dict
    """

    def __init__(
        self,
        feature_labels: dict[int, FeatureLabel],
        activation_threshold: float = 0.0,
    ) -> None:
        """
        Args:
            feature_labels:      Map from feature_id → FeatureLabel.
            activation_threshold: Minimum activation (exclusive) for a feature to be
                                  considered active. Default 0.0 matches TopK-SAE
                                  sparsity (all inactive features are exactly 0).
        """
        self.feature_labels = feature_labels
        self.activation_threshold = activation_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(
        self,
        activations: Union[np.ndarray, dict[int, float]],
    ) -> ClassifierResult:
        """
        Classify a single forward pass.

        Args:
            activations: Dense array of shape [dict_size] or sparse dict
                         {feature_id: activation}.  Features at or below
                         activation_threshold are treated as inactive.

        Returns:
            ClassifierResult with verdict, active labeled features, and confidence.
        """
        sparse = self._to_sparse(activations)
        active = self._find_active_labeled(sparse)
        active.sort(key=lambda f: f.activation * f.weight, reverse=True)

        flagged   = [f for f in active if f.safety_class == SafetyClass.FLAGGED]
        uncertain = [f for f in active if f.safety_class == SafetyClass.UNCERTAIN]
        safe      = [f for f in active if f.safety_class == SafetyClass.SAFE]

        flagged_score   = sum(f.activation * f.weight for f in flagged)
        uncertain_score = sum(f.activation * f.weight for f in uncertain)

        if flagged_score > 0:
            verdict = SafetyClass.FLAGGED
        elif uncertain_score > 0:
            verdict = SafetyClass.UNCERTAIN
        else:
            verdict = SafetyClass.SAFE

        confidence = self._confidence(
            verdict, flagged_score, uncertain_score, flagged, uncertain, safe
        )

        return ClassifierResult(
            verdict=verdict,
            active_features=active,
            confidence=confidence,
            n_flagged=len(flagged),
            n_uncertain=len(uncertain),
            n_safe=len(safe),
        )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _to_sparse(
        self, activations: Union[np.ndarray, dict[int, float]]
    ) -> dict[int, float]:
        if isinstance(activations, np.ndarray):
            return {
                int(i): float(v)
                for i, v in enumerate(activations)
                if v > self.activation_threshold
            }
        return {
            k: float(v)
            for k, v in activations.items()
            if v > self.activation_threshold
        }

    def _find_active_labeled(self, sparse: dict[int, float]) -> list[ActiveFeature]:
        features = []
        for fid, act in sparse.items():
            if fid in self.feature_labels:
                lbl = self.feature_labels[fid]
                features.append(
                    ActiveFeature(
                        feature_id=fid,
                        label=lbl.label,
                        safety_class=lbl.safety_class,
                        activation=act,
                        weight=lbl.weight,
                    )
                )
        return features

    def _confidence(
        self,
        verdict: SafetyClass,
        flagged_score: float,
        uncertain_score: float,
        flagged: list[ActiveFeature],
        uncertain: list[ActiveFeature],
        safe: list[ActiveFeature],
    ) -> float:
        if verdict == SafetyClass.SAFE:
            # 0.95 when no labeled features fired at all; 0.90 when only safe ones did
            return 0.95 if not safe else 0.90

        if verdict == SafetyClass.UNCERTAIN:
            return round(min(0.50 + 0.10 * len(uncertain), 0.90), 3)

        # FLAGGED
        base = min(0.60 + 0.10 * len(flagged), 0.95)
        # Mixed-evidence penalty: when uncertain weighted score is large relative
        # to flagged score the verdict is correct but less certain
        if uncertain_score > 0 and flagged_score > 0:
            total = flagged_score + uncertain_score
            mixed_penalty = (uncertain_score / total) * 0.20
            base = max(base - mixed_penalty, 0.50)
        return round(base, 3)
