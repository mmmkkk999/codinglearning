# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Dict, Any

from training.model.sam2 import SAM2Train
from training.loss_fns import SAM2DistillationLoss


class SAM2DistillationTrain(SAM2Train):
    """
    SAM2训练模型，支持知识蒸馏
    """
    
    def __init__(
        self,
        student_image_encoder,
        teacher_image_encoder,
        distillation_config: Dict[str, Any] = None,
        **kwargs
    ):
        # 初始化学生模型
        super().__init__(image_encoder=student_image_encoder, **kwargs)
        
        # 教师模型
        self.teacher_image_encoder = teacher_image_encoder
        
        # 冻结教师模型参数
        for param in self.teacher_image_encoder.parameters():
            param.requires_grad = False
        self.teacher_image_encoder.eval()
        
        # 蒸馏配置
        distillation_config = distillation_config or {}
        self.distillation_weight = distillation_config.get("weight", 1.0)
        self.temperature = distillation_config.get("temperature", 4.0)
        self.feature_weights = distillation_config.get("feature_weights", [1.0, 1.0, 1.0, 1.0])
        
        # 蒸馏损失
        self.distillation_loss = SAM2DistillationLoss(
            distillation_weight=self.distillation_weight,
            temperature=self.temperature,
            feature_weights=self.feature_weights
        )
        
    def forward(self, input_data):
        """
        前向传播，同时计算学生和教师模型的输出
        """
        # 学生模型前向传播
        student_outputs = super().forward(input_data)
        
        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_backbone_out = self.teacher_image_encoder.forward_image(input_data.flat_img_batch)
            teacher_outputs = self._prepare_backbone_features(teacher_backbone_out)
        
        # 将教师输出添加到学生输出中
        student_outputs["teacher_features"] = teacher_outputs
        
        return student_outputs
    
    def compute_distillation_loss(self, outputs, targets, original_loss):
        """
        计算蒸馏损失
        """
        student_features = outputs.get("backbone_fpn", [])
        teacher_features = outputs.get("teacher_features", [])
        
        # 计算蒸馏损失
        distill_loss = self.distillation_loss(
            student_features, 
            teacher_features
        )
        
        # 总损失
        total_loss = original_loss + self.distillation_weight * distill_loss
        
        return {
            "core_loss": total_loss,
            "distillation_loss": distill_loss,
            "original_loss": original_loss
        }


class SAM2FeatureDistillationTrain(SAM2Train):
    """
    仅对特征进行蒸馏的SAM2训练模型
    """
    
    def __init__(
        self,
        student_image_encoder,
        teacher_image_encoder,
        feature_distillation_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(image_encoder=student_image_encoder, **kwargs)
        
        # 教师模型
        self.teacher_image_encoder = teacher_image_encoder
        
        # 冻结教师模型
        for param in self.teacher_image_encoder.parameters():
            param.requires_grad = False
        self.teacher_image_encoder.eval()
        
        # 特征蒸馏权重
        self.feature_distillation_weight = feature_distillation_weight
        
    def forward_image(self, img_batch):
        """
        重写forward_image方法，同时获取学生和教师的特征
        """
        # 学生模型特征
        student_backbone_out = super().forward_image(img_batch)
        
        # 教师模型特征（不计算梯度）
        with torch.no_grad():
            teacher_backbone_out = self.teacher_image_encoder(img_batch)
        
        # 将教师特征添加到输出中
        student_backbone_out["teacher_backbone_fpn"] = teacher_backbone_out["backbone_fpn"]
        student_backbone_out["teacher_vision_pos_enc"] = teacher_backbone_out["vision_pos_enc"]
        
        return student_backbone_out
    
    def compute_feature_distillation_loss(self, student_features, teacher_features):
        """
        计算特征蒸馏损失
        """
        feature_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 归一化特征
            s_feat_norm = torch.nn.functional.normalize(s_feat, dim=1)
            t_feat_norm = torch.nn.functional.normalize(t_feat, dim=1)
            
            # MSE损失
            feat_loss = torch.nn.functional.mse_loss(s_feat_norm, t_feat_norm)
            feature_loss += feat_loss
            
        return feature_loss * self.feature_distillation_weight 