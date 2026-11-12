"""Titration policies: common interface, clinical baselines and RL adapters."""

from titraterl.policies.base import Policy
from titraterl.policies.rules import (
    ATLantus,
    FixedDose,
    Insight,
    ThresholdRule,
    ThresholdRuleParams,
    TreatToTarget,
    all_rule_policies,
    clinical_baselines,
)
from titraterl.policies.sb3 import SB3Policy

RULE_POLICIES = {
    "b0_fixed": FixedDose,
    "b1_atlantus": ATLantus,
    "b2_treat_to_target": TreatToTarget,
    "b3_insight": Insight,
    "b4_threshold": ThresholdRule,
}

__all__ = [
    "RULE_POLICIES",
    "ATLantus",
    "FixedDose",
    "Insight",
    "Policy",
    "SB3Policy",
    "ThresholdRule",
    "ThresholdRuleParams",
    "TreatToTarget",
    "all_rule_policies",
    "clinical_baselines",
]
