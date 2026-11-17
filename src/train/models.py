import segmentation_models_pytorch as smp
import torch.nn as nn
import yaml


def get_model(model_name: str, pretrained: bool = True, config_path: str = "configs/model.yaml") -> nn.Module:
    """
    Factory creating U-Net model variants according to configs/model.yaml:
    - 'r34': ResNet-34 encoder (~24.4M params)
    - 'mbv3s': MobileNetV3-Small encoder with shrunk decoder (~2.15M params)
    Output is raw logits (no Sigmoid in graph).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if model_name not in cfg:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from: {list(cfg.keys())}")

    m_cfg = cfg[model_name]
    weights = m_cfg.get("encoder_weights", "imagenet") if pretrained else None

    if model_name == "r34":
        model = smp.Unet(
            encoder_name=m_cfg["encoder_name"],
            encoder_weights=weights,
            in_channels=m_cfg.get("in_channels", 3),
            classes=m_cfg.get("classes", 1),
            activation=None,  # Logits output strictly
        )
    elif model_name == "mbv3s":
        encoder_name = m_cfg["encoder_name"]
        try:
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=m_cfg.get("in_channels", 3),
                classes=m_cfg.get("classes", 1),
                decoder_channels=m_cfg["decoder_channels"],
                activation=None,  # Logits output strictly
            )
        except Exception:
            fallback = m_cfg.get("encoder_name_fallback", "timm-mobilenetv3_small_100")
            print(f"Failed with {encoder_name}, falling back to {fallback}")
            model = smp.Unet(
                encoder_name=fallback,
                encoder_weights=weights,
                in_channels=m_cfg.get("in_channels", 3),
                classes=m_cfg.get("classes", 1),
                decoder_channels=m_cfg["decoder_channels"],
                activation=None,
            )
    else:
        raise NotImplementedError(f"Model {model_name} not configured.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created {model_name} model: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    if model_name == "mbv3s":
        assert 2_000_000 <= total_params <= 3_500_000, f"mbv3s parameters ({total_params}) outside target range (2M-3.5M)"

    return model
