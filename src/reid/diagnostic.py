"""Deterministic ReID Diagnostic Tool for auditing feature extraction, fusion, and decision gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.interfaces import BaseReID
from src.core.types import Embedding, Identity
from src.identity.manager import IdentityManager
from src.reid.quality import CropQualityEvaluator

logger = logging.getLogger(__name__)


@dataclass
class VectorStats:
    """Statistical summary of a single feature vector."""
    dim: int
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    l2_norm: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> VectorStats:
        flat = np.asarray(arr, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(flat))
        return cls(
            dim=int(flat.shape[0]),
            dtype=str(flat.dtype),
            min_val=float(np.min(flat)) if flat.size > 0 else 0.0,
            max_val=float(np.max(flat)) if flat.size > 0 else 0.0,
            mean_val=float(np.mean(flat)) if flat.size > 0 else 0.0,
            std_val=float(np.std(flat)) if flat.size > 0 else 0.0,
            l2_norm=norm,
        )

    def formatted(self, indent: str = "    ") -> str:
        return (
            f"{indent}dim     = {self.dim}\n"
            f"{indent}dtype   = {self.dtype}\n"
            f"{indent}min     = {self.min_val:+.6f}\n"
            f"{indent}max     = {self.max_val:+.6f}\n"
            f"{indent}mean    = {self.mean_val:+.6f}\n"
            f"{indent}std     = {self.std_val:.6f}\n"
            f"{indent}L2 norm = {self.l2_norm:.6f}"
        )


@dataclass
class GateResult:
    """Evaluation result for an individual gate."""
    name: str
    actual: float
    threshold: float
    operator: str  # ">=", "<=", "=="
    passed: bool
    description: str = ""

    def formatted(self, width: int = 28) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name:<{width}} = {self.actual:6.3f} (req {self.operator} {self.threshold:6.3f}) -> {status}"


@dataclass
class DecomposedComparisonResult:
    """Full decomposed comparison between a reference and a candidate crop."""
    # Stats
    ref_fused_stats: VectorStats
    ref_deep_stats: VectorStats
    ref_color_stats: VectorStats
    ref_upper_stats: VectorStats
    ref_lower_stats: VectorStats

    cand_fused_stats: VectorStats
    cand_deep_stats: VectorStats
    cand_color_stats: VectorStats
    cand_upper_stats: VectorStats
    cand_lower_stats: VectorStats

    # Similarities
    deep_sim: float
    color_sim: float
    upper_sim: float
    lower_sim: float
    fused_sim: float

    # Disagreement
    disagreement: float

    # Quality
    ref_quality_score: float
    ref_quality_reason: str
    cand_quality_score: float
    cand_quality_reason: str

    # Full gates against IdentityManager
    proto_sim: float
    best_ref_sim: float
    best_adaptive_sim: float
    top2_adaptive_mean: float
    candidate_score: float

    gates: List[GateResult] = field(default_factory=list)
    final_decision: str = "NO_MATCH"  # MATCH, NO_MATCH, AMBIGUOUS
    rejection_reasons: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("DECOMPOSED REID COMPONENT SIMILARITIES:")
        lines.append(f"  DeepSim                  = {self.deep_sim:.4f}")
        lines.append(f"  ColorSim                 = {self.color_sim:.4f}")
        lines.append(f"  UpperSim                 = {self.upper_sim:.4f}")
        lines.append(f"  LowerSim                 = {self.lower_sim:.4f}")
        lines.append(f"  FusedSim                 = {self.fused_sim:.4f}")
        lines.append(f"  ProtoSim                 = {self.proto_sim:.4f}")
        lines.append(f"  BestRefSim               = {self.best_ref_sim:.4f}")
        lines.append(f"  CandidateScore           = {self.candidate_score:.4f}")
        lines.append(f"  |DeepSim - ColorSim|     = {self.disagreement:.4f}")
        lines.append("-" * 60)
        lines.append("REJECTION GATES AUDIT:")
        for g in self.gates:
            lines.append("  " + g.formatted())
        lines.append("-" * 60)
        lines.append(f"FINAL DECISION: {self.final_decision}")
        if self.rejection_reasons:
            lines.append("REJECTION REASON(S):")
            for r in self.rejection_reasons:
                lines.append(f"  - {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ReIDDiagnostic:
    """
    Deterministic ReID diagnostic engine for rigorous auditing of
    embeddings, determinism, prototypes, and rejection gates.
    """

    def __init__(
        self,
        extractor: BaseReID,
        identity_manager: Optional[IdentityManager] = None,
        quality_evaluator: Optional[CropQualityEvaluator] = None,
    ) -> None:
        self.extractor = extractor
        self.quality = quality_evaluator or CropQualityEvaluator()
        self.identity_manager = identity_manager or IdentityManager(reid_extractor=extractor)

    def test_determinism(self, crop: np.ndarray, iterations: int = 3) -> Dict[str, Any]:
        """
        Runs the exact same crop through the extractor N times and measures cosine similarities.
        """
        if crop is None or crop.size == 0:
            raise ValueError("Crop cannot be empty")

        embeddings = []
        for i in range(iterations):
            fused, deep, color, upper, lower = self.extractor.extract_decomposed(crop)
            embeddings.append({
                "fused": fused,
                "deep": deep,
                "color": color,
                "upper": upper,
                "lower": lower,
            })

        pairwise_fused = []
        pairwise_deep = []
        pairwise_color = []

        for i in range(iterations):
            for j in range(i + 1, iterations):
                cos_f = embeddings[i]["fused"].cosine_similarity(embeddings[j]["fused"])
                cos_d = embeddings[i]["deep"].cosine_similarity(embeddings[j]["deep"])
                cos_c = embeddings[i]["color"].cosine_similarity(embeddings[j]["color"])
                pairwise_fused.append(((i + 1, j + 1), cos_f))
                pairwise_deep.append(((i + 1, j + 1), cos_d))
                pairwise_color.append(((i + 1, j + 1), cos_c))

        all_identical = all(np.isclose(sim, 1.0, atol=1e-5) for _, sim in pairwise_fused)

        return {
            "iterations": iterations,
            "pairwise_fused": pairwise_fused,
            "pairwise_deep": pairwise_deep,
            "pairwise_color": pairwise_color,
            "is_deterministic": all_identical,
        }

    def compare_crops(
        self,
        ref_crop: np.ndarray,
        cand_crop: np.ndarray,
        identity_id: str = "diag_target",
        second_cand_score: Optional[float] = None,
    ) -> DecomposedComparisonResult:
        """
        Extracts both crops, builds an isolated diagnostic identity,
        and computes all intermediate values and rejection gates.
        """
        # Quality evaluations
        ref_valid, ref_q_score, ref_q_reason = self.quality.evaluate(ref_crop)
        cand_valid, cand_q_score, cand_q_reason = self.quality.evaluate(cand_crop)

        # Decomposed extraction
        r_fused, r_deep, r_color, r_upper, r_lower = self.extractor.extract_decomposed(ref_crop)
        c_fused, c_deep, c_color, c_upper, c_lower = self.extractor.extract_decomposed(cand_crop)

        # Vector stats
        r_f_stats = VectorStats.from_array(r_fused.vector)
        r_d_stats = VectorStats.from_array(r_deep.vector)
        r_c_stats = VectorStats.from_array(r_color.vector)
        r_u_stats = VectorStats.from_array(r_upper.vector)
        r_l_stats = VectorStats.from_array(r_lower.vector)

        c_f_stats = VectorStats.from_array(c_fused.vector)
        c_d_stats = VectorStats.from_array(c_deep.vector)
        c_c_stats = VectorStats.from_array(c_color.vector)
        c_u_stats = VectorStats.from_array(c_upper.vector)
        c_l_stats = VectorStats.from_array(c_lower.vector)

        # Similarities
        deep_sim = float(r_deep.cosine_similarity(c_deep)) if r_deep.dim == c_deep.dim else 0.0
        color_sim = float(r_color.cosine_similarity(c_color)) if r_color.dim == c_color.dim else 0.0
        upper_sim = float(r_upper.cosine_similarity(c_upper)) if r_upper.dim == c_upper.dim else 0.0
        lower_sim = float(r_lower.cosine_similarity(c_lower)) if r_lower.dim == c_lower.dim else 0.0
        fused_sim = float(r_fused.cosine_similarity(c_fused)) if r_fused.dim == c_fused.dim else 0.0

        disagreement = abs(deep_sim - color_sim)

        # Set up an isolated IdentityManager evaluation
        self.identity_manager.clear()
        self.identity_manager.register_new_target(ref_crop, identity_id)

        eval_res = self.identity_manager.evaluate_candidate_crop(cand_crop, identity_id)

        # Build Gate Audit
        gates = []
        reasons = []

        # Gate 1: Reference Floor
        ref_anchor_score = max(eval_res.proto_sim, eval_res.best_ref_sim, deep_sim)
        g_ref = GateResult(
            name="RefAnchor (max(Proto,Ref,Deep))",
            actual=ref_anchor_score,
            threshold=self.identity_manager.reference_threshold,
            operator=">=",
            passed=(ref_anchor_score >= self.identity_manager.reference_threshold),
            description="Reference anchor appearance similarity",
        )
        gates.append(g_ref)
        if not g_ref.passed:
            reasons.append(f"RefAnchor {ref_anchor_score:.3f} < {self.identity_manager.reference_threshold:.2f}")

        # Gate 2: Candidate Score (Score-Level Consensus)
        g_cand = GateResult(
            name="CandidateScore (Consensus)",
            actual=eval_res.candidate_score,
            threshold=self.identity_manager.similarity_threshold,
            operator=">=",
            passed=(eval_res.candidate_score >= self.identity_manager.similarity_threshold),
            description="Score-level weighted multi-component consensus",
        )
        gates.append(g_cand)
        if not g_cand.passed:
            reasons.append(f"CandidateScore {eval_res.candidate_score:.3f} < {self.identity_manager.similarity_threshold:.2f}")

        # Gate 3: Upper Body Score
        g_upper = GateResult(
            name="UpperBody Sim",
            actual=upper_sim,
            threshold=0.50,
            operator=">=",
            passed=(upper_sim >= 0.50),
            description="Upper body appearance similarity",
        )
        gates.append(g_upper)

        # Gate 4: Deep Semantic Score
        g_deep = GateResult(
            name="Deep ReID Sim",
            actual=deep_sim,
            threshold=0.50,
            operator=">=",
            passed=(deep_sim >= 0.50),
            description="Deep Person-ReID semantic similarity",
        )
        gates.append(g_deep)

        # Gate 5: Quality Gate
        g_qual = GateResult(
            name="Crop Quality",
            actual=cand_q_score,
            threshold=0.35,
            operator=">=",
            passed=cand_valid,
            description=f"Quality valid ({cand_q_reason})",
        )
        gates.append(g_qual)
        if not g_qual.passed:
            reasons.append(f"Crop quality failed: {cand_q_reason}")

        # Gate 6: Margin Gate (if competing candidate exists)
        if second_cand_score is not None:
            margin = eval_res.candidate_score - second_cand_score
            g_margin = GateResult(
                name="Candidate Margin",
                actual=margin,
                threshold=self.identity_manager.min_margin,
                operator=">=",
                passed=(margin >= self.identity_manager.min_margin),
                description="Separation from 2nd best candidate",
            )
            gates.append(g_margin)
            if not g_margin.passed:
                reasons.append(f"Margin {margin:.3f} < {self.identity_manager.min_margin:.2f}")

        # Final Decision Determination (3-State Logic)
        if eval_res.is_match:
            if second_cand_score is not None and not gates[-1].passed:
                final_decision = "AMBIGUOUS"
            else:
                final_decision = "MATCH"
        elif g_cand.passed and not g_ref.passed:
            final_decision = "AMBIGUOUS"
        else:
            final_decision = "NO_MATCH"

        return DecomposedComparisonResult(
            ref_fused_stats=r_f_stats,
            ref_deep_stats=r_d_stats,
            ref_color_stats=r_c_stats,
            ref_upper_stats=r_u_stats,
            ref_lower_stats=r_l_stats,
            cand_fused_stats=c_f_stats,
            cand_deep_stats=c_d_stats,
            cand_color_stats=c_c_stats,
            cand_upper_stats=c_u_stats,
            cand_lower_stats=c_l_stats,
            deep_sim=deep_sim,
            color_sim=color_sim,
            upper_sim=upper_sim,
            lower_sim=lower_sim,
            fused_sim=fused_sim,
            disagreement=disagreement,
            ref_quality_score=ref_q_score,
            ref_quality_reason=ref_q_reason,
            cand_quality_score=cand_q_score,
            cand_quality_reason=cand_q_reason,
            proto_sim=eval_res.proto_sim,
            best_ref_sim=eval_res.best_ref_sim,
            best_adaptive_sim=eval_res.best_adaptive_sim,
            top2_adaptive_mean=eval_res.top2_adaptive_mean,
            candidate_score=eval_res.candidate_score,
            gates=gates,
            final_decision=final_decision,
            rejection_reasons=reasons,
        )
