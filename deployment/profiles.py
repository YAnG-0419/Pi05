"""Independent execution strategies and pi05 checkpoint adapters, plus legacy presets."""
from dataclasses import dataclass
import importlib
from pathlib import Path
from types import MappingProxyType
from fr3_wuji.paths import ROOT


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter: str
    checkpoint: Path
    port: int


MODELS = MappingProxyType({
    "19999": ModelSpec("19999", "joint54", ROOT / "checkpoints/pi05/19999", 8001),
    "30000": ModelSpec("30000", "joint64", ROOT / "checkpoints/pi05/30000", 8002),
})
STRATEGIES = ("slow", "normal")


@dataclass(frozen=True)
class Profile:
    name: str
    engine: str
    adapter: str
    checkpoint: Path
    port: int
    planning_profile: str = "slow54"

    @property
    def uri(self):
        return f"ws://127.0.0.1:{self.port}"

    def model_adapter(self):
        return importlib.import_module(f"deployment.models.{self.adapter}")


PROFILES = MappingProxyType({
    "slow54": Profile("slow54", "slow", "joint54", ROOT / "checkpoints/pi05/19999", 8001),
    "slow64": Profile("slow64", "slow", "joint64", ROOT / "checkpoints/pi05/30000", 8002, "slow64"),
    "normal": Profile("normal", "normal", "joint54", ROOT / "checkpoints/pi05/19999", 8001),
})


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown deployment profile {name!r}; choose {', '.join(PROFILES)}") from None


def compose_profile(strategy, model, *, planning_profile="slow54"):
    """Choose the model independently; planning limits never change implicitly with model dimensions."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown execution strategy: {strategy}")
    if model not in MODELS:
        raise ValueError(f"Unknown pi05 model: {model}")
    if planning_profile not in {"slow54", "slow64"}:
        raise ValueError(f"Unknown planning limits preset: {planning_profile}")
    if strategy == "normal" and planning_profile != "slow54":
        raise ValueError("Normal-speed acquisition retains the standard planning limits")
    spec = MODELS[model]
    return Profile(f"{strategy}-{model}", strategy, spec.adapter, spec.checkpoint, spec.port, planning_profile)
