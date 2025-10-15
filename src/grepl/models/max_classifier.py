"""Various straightforward classification models built on top of DinoV2 backbone.

- max over classification head on frame cls embeddings
- concat the avg of patch embeddings
- yet another transformer layer (considering all patch and cls embeddings) 
    to get an even better representation
"""

import torch
from torch import nn
# from grepl.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from grepl.layers.mlp import Mlp
from grepl.layers.attention import Attention
from grepl.layers.block import Block

class MaxFrameClassifier(nn.Module):
    """ A simple classifier that uses the DINOv2 model as a backbone.
    Linear probe on CLS embeddings
    """
    def __init__(self, backbone, num_classes, feature_dim=384, trainable_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.trainable_backbone = trainable_backbone
        if not trainable_backbone:
            self.backbone.eval()
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(in_features=feature_dim, out_features=num_classes)

    def forward(self, images):
        batch_size, n_frames_per_video, channels, height, width = images.shape
        images_flattened = images.flatten(0, 1)  # flatten batch and segments
        if self.trainable_backbone:
            features = self.backbone.forward_features(images_flattened)["x_norm_clstoken"]
        else:
            with torch.no_grad():
                features = self.backbone.forward_features(images_flattened)["x_norm_clstoken"]
        features = features.unflatten(0, (batch_size, n_frames_per_video))
        features = self.backbone.norm(features) # [batch, n_frames, feature_dim]
        logits = self.classifier(features) # [batch, n_frames, num_classes]
        max_logits = logits.max(dim=1)[0] # [batch, num_classes]
        return max_logits

class MaxFramePatchEmbeddingsClassifier(nn.Module):
    """In the forward pass, also consider the avg of patch embeddings."""
    
    def __init__(self, backbone, num_classes, feature_dim=384, trainable_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.trainable_backbone = trainable_backbone
        if not trainable_backbone:
            self.backbone.eval()
        self.feature_dim = feature_dim
        # because we concatenate cls and avg of patch features,
        # the classifier head has 2x the dimension
        self.classifier = nn.Linear(in_features=feature_dim * 2, out_features=num_classes)
    
    def forward(self, images):
        batch_size, n_frames_per_video, channels, height, width = images.shape
        images_flattened = images.flatten(0, 1)  # flatten batch and segments
        if self.trainable_backbone:
            forward_features = self.backbone.forward_features(images_flattened)
            cls_features = forward_features["x_norm_clstoken"]
            patch_features = forward_features["x_norm_patchtokens"].mean(dim=1) # avg over patches
            cls_features = self.backbone.norm(cls_features)
            patch_features = self.backbone.norm(patch_features)
            features = torch.cat([cls_features, patch_features], dim=-1) # [batch * n_frames, feature_dim * 2]
        else:
            with torch.no_grad():
                forward_features = self.backbone.forward_features(images_flattened)
                cls_features = forward_features["x_norm_clstoken"]
                patch_features = forward_features["x_norm_patchtokens"].mean(dim=1) # avg over patches
                cls_features = self.backbone.norm(cls_features)
                patch_features = self.backbone.norm(patch_features)
                features = torch.cat([cls_features, patch_features], dim=-1) # [batch * n_frames, feature_dim * 2]
        features = features.unflatten(0, [batch_size, n_frames_per_video]) # [batch, n_frames, feature_dim * 2]
        logits = self.classifier(features) # [batch, n_frames, num_classes]
        max_logits = logits.max(dim=1)[0] # [batch, num_classes]
        return max_logits

class MaxFrameTransformerHeadClassifier(nn.Module):
    """Add another transformer block"""
    
    def __init__(self, backbone, num_classes, feature_dim=384, n_blocks=2, trainable_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.trainable_backbone = trainable_backbone
        if not trainable_backbone:
            self.backbone.eval()
        self.feature_dim = feature_dim
        # add a transformer block before the final classifier
        blocks_list = [
            Block(
                dim=feature_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1,
            )
            for _ in range(n_blocks)
        ]
        # self.transformer_blocks = nn.Sequential(*blocks_list)
        self.transformer_blocks = nn.ModuleList(blocks_list)
        self.classifier = nn.Linear(in_features=feature_dim, out_features=num_classes, bias=True)
        # learnable parameters for the classifier head
        # self.feature_frame_decay = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.logit_frame_decay = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, images):
        batch_size, n_frames_per_video, channels, height, width = images.shape
        images_flattened = images.flatten(0, 1)  # flatten batch and segments
        if self.trainable_backbone:
            x = self.backbone.prepare_tokens_with_masks(images_flattened)
            for blk in self.backbone.blocks:
                x = blk(x)
        else:
            with torch.no_grad():
                x = self.backbone.prepare_tokens_with_masks(images_flattened)
                for blk in self.backbone.blocks:
                    x = blk(x)
        for blk in self.transformer_blocks:
            x = blk(x)
        x_norm = self.backbone.norm(x)
        x_cls = x_norm[:, 0] # CLS token
        features = x_cls.unflatten(0, [batch_size, n_frames_per_video]) # [batch, n_frames, feature_dim]
        # feature_frame_scale = self.feature_frame_decay * torch.arange(n_frames_per_video, device=features.device).unsqueeze(0).unsqueeze(-1) 
        # feature_frame_scale = feature_frame_scale.exp()
        # features *= feature_frame_scale
        logits = self.classifier(features) # [batch, n_frames, num_classes]
        logit_scale = self.logit_frame_decay * torch.arange(n_frames_per_video, device=logits.device).unsqueeze(0).unsqueeze(-1) 
        logit_scale = logit_scale.exp()
        logits = logits * logit_scale
        max_logits = logits.max(dim=1)[0]
        return max_logits