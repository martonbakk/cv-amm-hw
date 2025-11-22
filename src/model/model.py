from typing import  Tuple
import torch
import torch.nn as nn
from src.config.configuration import CLASS_NUM
import timm
import logging


class BackboneOnly(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        logging.info("Creating BackboneOnly model.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"BackBone using device: {self.device}")

        self.backbone: nn.Module = backbone.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).to(self.device)
            

class TwoHeadConvNeXtV2(nn.Module):
    """ConvNeXt V2 backbone + classification heads (binary + multi-class)
    """
    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae",
        pretrained: bool = True,
        num_multi_classes: int = CLASS_NUM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            self.feature_dim: int = self.__backbone.num_features # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            raise ValueError("Could not determine feature dimension from backbone.")

        self.__binary_head: nn.Sequential = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # 1 logit
        ).to(self.device)

        self.__multi_head: nn.Sequential = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_multi_classes),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Visszaad: (binary_logit, multi_logits)
        binary_logit: [B, 1]
        multi_logits: [B, C]
        """
        feats: torch.Tensor = self.__backbone(x).to(self.device)
        # backbone may return shape [B, C]
        b_logit: torch.Tensor = self.__binary_head(feats).to(self.device)
        m_logits: torch.Tensor = self.__multi_head(feats).to(self.device)
        return b_logit.squeeze(1), m_logits

    def freeze_backbone(self):
        """Befagyasztja a backbone paramétereit (nem tanulódnak)."""
        for p in self.__backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.__backbone.parameters():
            p.requires_grad = True

    @property
    def backbone(self) -> nn.Module:
        return BackboneOnly(self.__backbone)
    
    @property
    def binary_head(self) -> nn.Module:
        return self.__binary_head
    
    @property
    def species_head(self) -> nn.Module:
        return self.__multi_head
