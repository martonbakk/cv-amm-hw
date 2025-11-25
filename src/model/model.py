from typing import Tuple
import torch
import torch.nn as nn
from src.config.configuration import CLASS_NUM, MODEL_NAME
import timm
import logging


class TwoHeadConvNeXtV2(nn.Module):
    """ConvNeXt V2 backbone + classification heads (binary + multi-class)"""

    def __init__(
        self,
        backbone_name: str = MODEL_NAME,
        pretrained: bool = True,
        num_multi_classes: int = CLASS_NUM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logging.warning(
                "CPU device detected. Inference or Training on CPU may be very slow. "
                "Consider using a GPU for better performance."
            )
        logging.info(f"Using device: {self.device}")
        logging.info(f"Creating TwoHeadConvNeXtV2 with backbone {backbone_name}")
        # timm model with num_classes=0 returns a model whose forward gives feature vector
        # some timm versions support features_only=True; here we use num_classes=0 and get out_features
        self.__backbone: nn.Module = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.1,
        )
        self.__backbone.to(self.device)

        # find feature dim (backbone.num_features is commonly set)
        try:
            self.feature_dim: int = (
                self.__backbone.num_features
            )  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            raise ValueError("Could not determine feature dimension from backbone.")

        self.__binary_head: nn.Sequential = (
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),  # 1 logit
            )
            .to(self.device)
            .apply(init_head_weights)
        )

        self.__multi_head: nn.Sequential = (
            nn.Sequential(
                nn.Linear(self.feature_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, CLASS_NUM),
            )
            .to(self.device)
            .apply(init_head_weights)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass a batch through the model.

        Args:
            x: Input tensor of shape [B, C, H, W] where B is batch size,
               C is channels (3 for RGB), H and W are height and width.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - binary_logit: Binary classification logits of shape [B]
                - multi_logits: Multi-class classification logits of shape [B, num_classes]
        """
        feats: torch.Tensor = self.__backbone(x).to(self.device)
        # backbone may return shape [B, C]
        b_logit: torch.Tensor = self.__binary_head(feats).to(self.device)
        m_logits: torch.Tensor = self.__multi_head(feats).to(self.device)
        return b_logit.squeeze(1), m_logits

    def freeze_backbone(self):
        """Freeze the backbone parameters."""
        for p in self.__backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        for p in self.__backbone.parameters():
            p.requires_grad = True

    @property
    def backbone(self) -> nn.Module:
        return self.__backbone

    @property
    def binary_head(self) -> nn.Module:
        return self.__binary_head

    @property
    def species_head(self) -> nn.Module:
        return self.__multi_head


def init_head_weights(m: nn.Module):
    """
    Weight initialization function for classification heads.
    Only affects Linear layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
