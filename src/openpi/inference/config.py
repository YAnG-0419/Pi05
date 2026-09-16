"""FR3/Wuji inference configuration, independent of training and dataset loaders."""

import dataclasses
import pathlib
from typing import Any

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import tokenizer
from openpi.policies import fr3_wuji_policy


@dataclasses.dataclass(frozen=True)
class DataConfig:
    asset_id: str = "fr3_wuji/tomato"
    data_transforms: transforms.Group = dataclasses.field(default_factory=transforms.Group)
    model_transforms: transforms.Group = dataclasses.field(default_factory=transforms.Group)
    use_quantile_norm: bool = True


@dataclasses.dataclass(frozen=True)
class Fr3WujiTransforms:
    # 19999 uses arm deltas; the 30000 adapter explicitly disables this transform.
    extra_delta_transform: bool = True

    def create(self, assets_dirs: pathlib.Path, model_config: pi0_config.Pi0Config) -> DataConfig:
        del assets_dirs  # Normalization must come from the selected checkpoint.
        data_transforms = transforms.Group(
            inputs=[fr3_wuji_policy.Fr3WujiInputs(model_type=model_config.model_type)],
            outputs=[fr3_wuji_policy.Fr3WujiOutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(7, -20, 7, -20)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return DataConfig(
            data_transforms=data_transforms,
            model_transforms=transforms.Group(inputs=[
                transforms.InjectDefaultPrompt(None),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                transforms.PadStatesAndActions(model_config.action_dim),
            ]),
        )


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    name: str = "pi05_fr3_wuji"
    model: pi0_config.Pi0Config = dataclasses.field(default_factory=lambda: pi0_config.Pi0Config(
        pi05=True,
        action_dim=54,
        action_horizon=50,
        max_token_len=256,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ))
    data: Fr3WujiTransforms = dataclasses.field(default_factory=Fr3WujiTransforms)
    policy_metadata: dict[str, Any] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[3] / "assets" / self.name


def get_config(name: str) -> InferenceConfig:
    if name != "pi05_fr3_wuji":
        raise ValueError(f"Unsupported deployment config: {name}")
    return InferenceConfig()
