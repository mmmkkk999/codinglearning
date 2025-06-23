#!/usr/bin/env python3
"""
优化的Hiera-T知识蒸馏训练脚本
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.trainer import Trainer
from training.utils.logger import setup_logging
from training.utils.distributed import setup_distributed
from training.utils.checkpoint_utils import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Hiera-T知识蒸馏训练")
    parser.add_argument(
        "--config", 
        type=str, 
        default="sam2/configs/sam2_distillation/sam2_distillation_hiera_t_optimal.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--teacher_checkpoint", 
        type=str, 
        default="checkpoints/sam2.1_hiera_t.pt",
        help="教师模型检查点路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/hiera_t_distillation",
        help="输出目录"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True,
        help="数据集根目录"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="数据加载器工作进程数"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=80,
        help="训练轮数"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1.5e-4,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="权重衰减"
    )
    parser.add_argument(
        "--distillation_weight", 
        type=float, 
        default=1.5,
        help="蒸馏损失权重"
    )
    parser.add_argument(
        "--warmup_epochs", 
        type=int, 
        default=5,
        help="预热轮数"
    )
    parser.add_argument(
        "--save_freq", 
        type=int, 
        default=10,
        help="保存频率（轮数）"
    )
    parser.add_argument(
        "--eval_freq", 
        type=int, 
        default=5,
        help="评估频率（轮数）"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="恢复训练的检查点路径"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0,
        help="本地GPU排名"
    )
    parser.add_argument(
        "--distributed", 
        action="store_true",
        help="是否使用分布式训练"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置分布式训练
    if args.distributed:
        setup_distributed(args.local_rank)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 更新配置
    config.teacher_checkpoint = args.teacher_checkpoint
    config.trainer.max_epochs = args.epochs
    config.optim.optimizer.lr = args.lr
    config.optim.optimizer.weight_decay = args.weight_decay
    config.model.feature_distillation_weight = args.distillation_weight
    config.data.train.data_root = args.data_root
    config.data.train.batch_size = args.batch_size
    config.data.train.num_workers = args.num_workers
    
    # 设置日志
    logger = setup_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        rank=args.local_rank if args.distributed else 0
    )
    
    logger.info("=== Hiera-T 知识蒸馏训练开始 ===")
    logger.info(f"配置: {args.config}")
    logger.info(f"教师模型: {args.teacher_checkpoint}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"蒸馏权重: {args.distillation_weight}")
    logger.info(f"训练轮数: {args.epochs}")
    
    # 创建训练器
    trainer = Trainer.from_config(config)
    
    # 恢复训练
    if args.resume:
        logger.info(f"恢复训练: {args.resume}")
        load_checkpoint(trainer.model, args.resume, strict=False)
    
    # 开始训练
    try:
        trainer.fit()
        logger.info("训练完成！")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pt")
    trainer.save_checkpoint(final_checkpoint_path)
    logger.info(f"最终模型已保存: {final_checkpoint_path}")

if __name__ == "__main__":
    main() 