"""
Training Stabilization Utilities for DiffBinaural
学習安定化ユーティリティ（既存システムへの追加機能）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any

class GradientStabilizer:
    """
    勾配安定化ユーティリティ（既存の学習ループに追加可能）
    """
    
    def __init__(self, clip_norm: float = 1.0, clip_value: Optional[float] = None):
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.grad_norms = []
        
    def stabilize_gradients(self, model: nn.Module) -> Dict[str, float]:
        """勾配をクリップして安定化"""
        # 勾配ノルムを計算
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        self.grad_norms.append(total_norm)
        
        # 勾配クリップ
        if self.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        
        return {
            'grad_norm': total_norm,
            'param_count': param_count,
            'avg_grad_norm': np.mean(self.grad_norms[-100:])  # 直近100ステップの平均
        }

class LossStabilizer:
    """
    損失値の安定化と監視（既存システムに追加可能）
    """
    
    def __init__(self, smoothing_factor: float = 0.99, anomaly_threshold: float = 10.0):
        self.smoothing_factor = smoothing_factor
        self.anomaly_threshold = anomaly_threshold
        self.loss_history = []
        self.smoothed_loss = None
        
    def update_and_check(self, loss_value: float) -> Dict[str, Any]:
        """損失値を更新して異常値をチェック"""
        self.loss_history.append(loss_value)
        
        # 指数移動平均で平滑化
        if self.smoothed_loss is None:
            self.smoothed_loss = loss_value
        else:
            self.smoothed_loss = (self.smoothing_factor * self.smoothed_loss + 
                                (1 - self.smoothing_factor) * loss_value)
        
        # 異常値検出
        is_anomaly = False
        if len(self.loss_history) > 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if loss_value > recent_avg * self.anomaly_threshold:
                is_anomaly = True
                warnings.warn(f"Anomalous loss detected: {loss_value:.6f} "
                            f"(recent avg: {recent_avg:.6f})")
        
        return {
            'loss': loss_value,
            'smoothed_loss': self.smoothed_loss,
            'is_anomaly': is_anomaly,
            'loss_std': np.std(self.loss_history[-100:]) if len(self.loss_history) > 10 else 0.0
        }

class LearningRateStabilizer:
    """
    学習率の動的調整（既存のオプティマイザーと組み合わせ可能）
    """
    
    def __init__(self, patience: int = 10, factor: float = 0.5, min_lr: float = 1e-7):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')
        
    def step(self, val_loss: float, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """検証損失に基づいて学習率を調整"""
        lr_reduced = False
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                if current_lr > self.min_lr:
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    lr_reduced = True
                    print(f"Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")
                self.wait = 0
        
        return {
            'lr_reduced': lr_reduced,
            'current_lr': optimizer.param_groups[0]['lr'],
            'best_loss': self.best_loss,
            'patience_wait': self.wait
        }

class MemoryStabilizer:
    """
    GPUメモリ使用量の監視と最適化
    """
    
    @staticmethod
    def clear_cache():
        """GPUキャッシュをクリア"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """メモリ使用情報を取得"""
        if not torch.cuda.is_available():
            return {'memory_used': 0, 'memory_total': 0}
        
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_total = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return {
            'memory_used': memory_used,
            'memory_total': memory_total,
            'memory_percent': (memory_used / memory_total * 100) if memory_total > 0 else 0
        }
    
    @staticmethod
    def optimize_memory_usage(threshold_percent: float = 80.0):
        """メモリ使用率が閾値を超えた場合に最適化"""
        memory_info = MemoryStabilizer.get_memory_info()
        
        if memory_info['memory_percent'] > threshold_percent:
            MemoryStabilizer.clear_cache()
            print(f"Memory usage was {memory_info['memory_percent']:.1f}%, cleared cache")
            return True
        return False

class TrainingStabilizer:
    """
    包括的な学習安定化マネージャー（既存システムに簡単に統合可能）
    """
    
    def __init__(self, 
                 grad_clip_norm: float = 1.0,
                 loss_smoothing: float = 0.99,
                 lr_patience: int = 10,
                 memory_threshold: float = 80.0):
        
        self.grad_stabilizer = GradientStabilizer(grad_clip_norm)
        self.loss_stabilizer = LossStabilizer(loss_smoothing)
        self.lr_stabilizer = LearningRateStabilizer(lr_patience)
        self.memory_threshold = memory_threshold
        
        self.step_count = 0
        
    def training_step(self, 
                     model: nn.Module, 
                     optimizer: optim.Optimizer, 
                     loss_value: float) -> Dict[str, Any]:
        """学習ステップ後の安定化処理"""
        self.step_count += 1
        
        # 勾配安定化
        grad_info = self.grad_stabilizer.stabilize_gradients(model)
        
        # 損失安定化
        loss_info = self.loss_stabilizer.update_and_check(loss_value)
        
        # メモリ最適化（定期的に実行）
        memory_optimized = False
        if self.step_count % 100 == 0:
            memory_optimized = MemoryStabilizer.optimize_memory_usage(self.memory_threshold)
        
        return {
            **grad_info,
            **loss_info,
            'memory_optimized': memory_optimized,
            'step_count': self.step_count
        }
    
    def validation_step(self, val_loss: float, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """バリデーション後の学習率調整"""
        return self.lr_stabilizer.step(val_loss, optimizer)

class ModelCheckpointer:
    """
    モデルチェックポイントの安全な保存・復元
    """
    
    def __init__(self, checkpoint_dir: str, keep_best: int = 3):
        import os
        self.checkpoint_dir = checkpoint_dir
        self.keep_best = keep_best
        self.best_losses = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       loss: float,
                       extra_data: Optional[Dict] = None) -> str:
        """チェックポイントを安全に保存"""
        import os
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'step_count': getattr(self, 'step_count', 0)
        }
        
        if extra_data:
            checkpoint_data.update(extra_data)
        
        # ファイル名作成
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}_loss_{loss:.6f}.pth'
        )
        
        # 安全に保存
        temp_path = checkpoint_path + '.tmp'
        try:
            torch.save(checkpoint_data, temp_path)
            os.rename(temp_path, checkpoint_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
        
        # ベストモデル管理
        self.best_losses.append((loss, checkpoint_path))
        self.best_losses.sort(key=lambda x: x[0])
        
        # 古いチェックポイント削除
        if len(self.best_losses) > self.keep_best:
            _, old_path = self.best_losses.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
        
        return checkpoint_path

# 既存システムとの統合用ヘルパー関数
def create_stabilized_training_loop(model: nn.Module, 
                                  optimizer: optim.Optimizer,
                                  checkpoint_dir: str = './checkpoints_stabilized'):
    """既存の学習ループに安定化機能を追加"""
    stabilizer = TrainingStabilizer()
    checkpointer = ModelCheckpointer(checkpoint_dir)
    
    return stabilizer, checkpointer

def apply_training_stabilization(loss_value: float, 
                               model: nn.Module, 
                               optimizer: optim.Optimizer) -> Dict[str, Any]:
    """単発の安定化処理（既存コードに1行追加）"""
    stabilizer = TrainingStabilizer()
    return stabilizer.training_step(model, optimizer, loss_value) 