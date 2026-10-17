"""CPU-only checks for the isolated 30000v2 fast entry."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy.fr3_wuji_30000 import fast
from deploy.fr3_wuji_30000 import serve
from openpi.training import config


def test_fast_defaults_select_v2_and_port_8002():
    args = fast.with_defaults(["--check"])
    assert args == [
        "--check",
        "--checkpoint",
        str(ROOT / "checkpoints/30000v2"),
        "--uri",
        "ws://127.0.0.1:8002",
        "--arm-motion-limits",
        "legacy",
    ]


def test_explicit_values_are_preserved():
    args = fast.with_defaults(["--check", "--checkpoint=/tmp/model", "--uri=ws://127.0.0.1:9000"])
    assert args == [
        "--check",
        "--checkpoint=/tmp/model",
        "--uri=ws://127.0.0.1:9000",
        "--arm-motion-limits",
        "legacy",
    ]


def test_explicit_disabled_motion_policy_is_preserved():
    args = fast.with_defaults(["--check", "--arm-motion-limits=disabled"])
    assert "--arm-motion-limits=disabled" in args
    assert args.count("--arm-motion-limits") == 0


def test_v2_contract_is_fast_pipeline_compatible():
    contract, _ = serve.checkpoint_contract(ROOT / "checkpoints/30000v2")
    assert contract["checkpoint"] == str((ROOT / "checkpoints/30000v2").resolve())
    assert contract["model_action_dim"] == 64
    assert contract["action_dim"] == 54
    assert contract["action_horizon"] == 50
    assert contract["action_representation"] == "absolute_joint_positions"


def test_v2_uses_full_finetune_model_variants():
    base = config.get_config("pi05_fr3_wuji").model
    assert base.paligemma_variant == "gemma_2b_lora"
    assert base.action_expert_variant == "gemma_300m_lora"
    full = serve.full_finetune_model(base)
    assert full.paligemma_variant == "gemma_2b"
    assert full.action_expert_variant == "gemma_300m"
    assert full.action_dim == 64
