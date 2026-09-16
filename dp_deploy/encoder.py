"""Independent RGB and one-channel depth ResNets for official image DP policies."""

import torch
from torch import nn
from torchvision.models import resnet18
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class RGBDObsEncoder(ModuleAttrMixin):
    def __init__(self, shape_meta, use_group_norm=True, resize_shape=None):
        super().__init__()
        self.shapes = {
            key: tuple(spec["shape"]) for key, spec in shape_meta["obs"].items()
        }
        self.resize_shape = tuple(resize_shape) if resize_shape is not None else None
        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or min(self.resize_shape) < 1
        ):
            raise ValueError("resize_shape must contain positive height and width")
        self.kinds = {
            key: spec.get("type", "low_dim") for key, spec in shape_meta["obs"].items()
        }
        self.models = nn.ModuleDict()
        self.lowdim = []
        self.feature_dim = 0
        for key, spec in sorted(shape_meta["obs"].items()):
            kind = spec.get("type", "low_dim")
            if kind in ("rgb", "depth"):
                channels = 3 if kind == "rgb" else 1
                if len(spec["shape"]) != 3 or spec["shape"][0] != channels:
                    raise ValueError(f"{key}: invalid {kind} shape")
                model = resnet18(weights=None)
                if channels == 1:
                    model.conv1 = nn.Conv2d(
                        1, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                model.fc = nn.Identity()
                if use_group_norm:
                    model = replace_submodules(
                        model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            x.num_features // 16, x.num_features
                        ),
                    )
                self.models[key] = model
                self.feature_dim += 512
            elif kind == "low_dim" and len(spec["shape"]) == 1:
                self.lowdim.append(key)
                self.feature_dim += spec["shape"][0]
            else:
                raise ValueError(f"Unsupported observation: {key}: {kind}")

    def output_shape(self):
        return (self.feature_dim,)

    def forward(self, obs):
        for key, shape in self.shapes.items():
            if tuple(obs[key].shape[1:]) != shape:
                raise ValueError(
                    f"{key}: expected B,{shape}, got {tuple(obs[key].shape)}"
                )
        features = []
        for key, model in self.models.items():
            image = obs[key]
            if self.resize_shape is not None:
                if self.kinds[key] == "depth":
                    image = torch.nn.functional.interpolate(
                        image, size=self.resize_shape, mode="nearest"
                    )
                else:
                    image = torch.nn.functional.interpolate(
                        image,
                        size=self.resize_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
            features.append(model(image))
        features += [obs[key] for key in self.lowdim]
        return torch.cat(features, dim=-1)
