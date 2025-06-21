#!/usr/bin/env python3
"""
SAM2 Image Encoder 蒸馏训练脚本
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.trainer import Trainer


def load_teacher_checkpoint(teacher_model, checkpoint_path):
    """
    加载教师模型的检查点
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"教师模型检查点不存在: {checkpoint_path}")
    
    print(f"加载教师模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载教师模型权重
    if 'model' in checkpoint:
        teacher_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        teacher_model.load_state_dict(checkpoint, strict=False)
    
    print("教师模型加载成功")
    return teacher_model


@hydra.main(version_base=None, config_path="../../sam2/configs/sam2_distillation", config_name="sam2_distillation_hiera_s")
def main(cfg: DictConfig):
    """
    主训练函数
    """
    print("=" * 50)
    print("SAM2 Image Encoder 蒸馏训练")
    print("=" * 50)
    
    # 打印配置
    print("配置信息:")
    print(OmegaConf.to_yaml(cfg))
    
    # 检查教师模型检查点
    teacher_checkpoint = cfg.get("teacher_checkpoint", None)
    if teacher_checkpoint is None:
        print("警告: 未指定教师模型检查点路径")
        print("请确保教师模型已经正确初始化")
    else:
        print(f"教师模型检查点: {teacher_checkpoint}")
    
    # 创建训练器
    trainer = Trainer(**cfg.trainer)
    
    # 如果指定了教师模型检查点，加载它
    if teacher_checkpoint and hasattr(trainer.model, 'teacher_image_encoder'):
        load_teacher_checkpoint(
            trainer.model.teacher_image_encoder, 
            teacher_checkpoint
        )
    
    # 开始训练
    print("开始训练...")
    trainer.run()
    
    print("训练完成!")


if __name__ == "__main__":
    main()