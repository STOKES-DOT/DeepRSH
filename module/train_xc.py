import torch
import torch.nn as nn
from xc_loss import loss
from DeepsRSHXC import DeepsRSHXC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import numpy as np
from tqdm import tqdm
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Mol2Dataset(Dataset):
    def __init__(self, mol2_directory):
        """
        分子数据集 - 所有mol2文件在同一个文件夹
        Args:
            mol2_directory: 包含mol2文件的目录
        """
        self.mol2_files = glob.glob(os.path.join(mol2_directory, "*.mol2"))
        if len(self.mol2_files) == 0:
            raise ValueError(f"No mol2 files found in {mol2_directory}")
        
        logger.info(f"Found {len(self.mol2_files)} mol2 files")
        
    def __len__(self):
        return len(self.mol2_files)
    
    def __getitem__(self, idx):
        return self.mol2_files[idx]

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # 训练配置
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.epochs = self.config.get('epochs', 100)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = self.config.get('save_dir', 'checkpoints')
        self.log_interval = self.config.get('log_interval', 10)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=self.config.get('lr_patience', 10),
            factor=self.config.get('lr_factor', 0.5),
            verbose=True
        )
        
        # 损失函数
        self.criterion = loss()
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': [],
            'best_parameters': None
        }
        
        # 可视化设置
        self.viz_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Number of trainable parameters: {self.count_parameters()}")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        epoch_start_time = datetime.now()
        
        for batch_idx, mol2_files in enumerate(progress_bar):
            batch_loss = 0
            batch_count = 0
            
            for mol2_file in mol2_files:
                try:
                    # 前向传播
                    xc_functional = self.model(mol2_file)
                    
                    # 计算损失
                    current_loss = self.criterion(xc_functional, mol2_file)
                    
                    # 累加损失
                    batch_loss += current_loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing {mol2_file}: {e}")
                    continue
            
            if batch_count == 0:
                continue
                
            # 平均损失
            avg_batch_loss = batch_loss / batch_count
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_tensor = torch.tensor(avg_batch_loss, requires_grad=True)
            loss_tensor.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += avg_batch_loss
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{avg_batch_loss:.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}'
            })
            
            # 记录日志
            if batch_idx % self.log_interval == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_batch_loss:.6f}')
        
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        self.train_history['epoch_times'].append(epoch_time)
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def validate(self, epoch):
        if self.val_loader is None:
            return float('inf')
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # 记录验证过程中的参数
        val_parameters = []
        
        with torch.no_grad():
            for mol2_files in tqdm(self.val_loader, desc=f'Validation Epoch {epoch}'):
                batch_loss = 0
                batch_count = 0
                
                for mol2_file in mol2_files:
                    try:
                        # 这里需要修改以获取参数
                        # 由于原始模型只返回xc_functional字符串，我们需要修改模型以同时返回参数
                        xc_functional = self.model(mol2_file)
                        current_loss = self.criterion(xc_functional, mol2_file)
                        batch_loss += current_loss.item()
                        batch_count += 1
                    except Exception as e:
                        logger.warning(f"Validation error processing {mol2_file}: {e}")
                        continue
                
                if batch_count > 0:
                    total_loss += batch_loss / batch_count
                    num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def train(self):
        best_val_loss = float('inf')
        start_time = datetime.now()
        
        logger.info("Starting training...")
        
        for epoch in range(1, self.epochs + 1):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)
                logger.info(f"New best model saved with val_loss: {val_loss:.6f}")
            
            # 定期保存检查点和可视化
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
                self.visualize_training(epoch)
            
            # 记录epoch结果
            logger.info(f"Epoch {epoch}/{self.epochs}: "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                       f"Time: {self.train_history['epoch_times'][-1]:.2f}s")
        
        # 保存最终模型和训练历史
        self.save_checkpoint(self.epochs, final=True)
        self.save_training_history()
        self.visualize_training(self.epochs, final=True)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {total_time/60:.2f} minutes!")
    
    def save_checkpoint(self, epoch, best=False, final=False):
        if best:
            filename = f"best_model.pth"
        elif final:
            filename = f"final_model_epoch_{epoch}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'config': self.config
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def save_training_history(self):
        history_file = os.path.join(self.save_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            json_ready_history = {}
            for key, value in self.train_history.items():
                if key == 'best_parameters' and value is not None:
                    json_ready_history[key] = value
                else:
                    json_ready_history[key] = [float(x) if isinstance(x, (np.floating, float)) else x for x in value]
            json.dump(json_ready_history, f, indent=2)
        
        # 也保存为npy格式用于绘图
        np.save(os.path.join(self.save_dir, 'training_history.npy'), self.train_history)
    
    def visualize_training(self, epoch, final=False):
        """可视化训练过程"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 训练和验证损失
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if self.train_history['val_loss'] and self.train_history['val_loss'][0] != float('inf'):
            axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习率
        axes[0, 1].plot(epochs, self.train_history['learning_rates'], 'g-', linewidth=2)
        axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 每个epoch的训练时间
        if len(self.train_history['epoch_times']) > 0:
            axes[1, 0].plot(epochs, self.train_history['epoch_times'], 'purple', linewidth=2)
            axes[1, 0].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 损失分布直方图 (最后几个epoch)
        if len(self.train_history['train_loss']) >= 5:
            recent_losses = self.train_history['train_loss'][-5:]
            axes[1, 1].hist(recent_losses, bins=10, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_title('Recent Training Loss Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if final:
            filename = 'training_summary_final.png'
        else:
            filename = f'training_progress_epoch_{epoch}.png'
        
        plt.savefig(os.path.join(self.viz_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建损失对比图（如果验证损失可用）
        if self.train_history['val_loss'] and self.train_history['val_loss'][0] != float('inf'):
            self._create_comparison_plot(epoch, final)
    
    def _create_comparison_plot(self, epoch, final=False):
        """创建训练和验证损失的对比图"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        plt.plot(epochs, self.train_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        # 标记最佳验证点
        best_epoch = np.argmin(self.train_history['val_loss']) + 1
        best_loss = min(self.train_history['val_loss'])
        plt.plot(best_epoch, best_loss, 'ro', markersize=8, label=f'Best Val Loss: {best_loss:.4f}')
        
        plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if final:
            filename = 'train_val_comparison_final.png'
        else:
            filename = f'train_val_comparison_epoch_{epoch}.png'
        
        plt.savefig(os.path.join(self.viz_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def collate_fn(batch):
    """自定义collate函数处理mol2文件路径"""
    return batch

def create_data_loaders(mol2_directory, batch_size=4, val_ratio=0.2, num_workers=4):
    """创建训练和验证数据加载器"""
    # 创建完整数据集
    full_dataset = Mol2Dataset(mol2_directory)
    
    # 计算分割大小
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子以便复现
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def plot_molecule_distribution(mol2_directory, save_path):
    """绘制分子数据集中分子大小的分布"""
    mol2_files = glob.glob(os.path.join(mol2_directory, "*.mol2"))
    
    atom_counts = []
    for mol2_file in mol2_files:
        try:
            with open(mol2_file, 'r') as f:
                lines = f.readlines()
            
            atom_section = False
            count = 0
            for line in lines:
                if line.startswith('@<TRIPOS>ATOM'):
                    atom_section = True
                    continue
                elif line.startswith('@<TRIPOS>'):
                    atom_section = False
                    continue
                if atom_section and line.strip():
                    count += 1
            
            atom_counts.append(count)
        except:
            continue
    
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Molecule Sizes (Atom Count)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Molecule size distribution saved to {save_path}")
    logger.info(f"Average atoms per molecule: {np.mean(atom_counts):.2f}")
    logger.info(f"Min atoms: {min(atom_counts)}, Max atoms: {max(atom_counts)}")

def main():
    # 配置参数
    config = {
        'learning_rate': 1e-4,
        'epochs': 100,
        'batch_size': 4,  # 由于量子化学计算昂贵，使用小批量
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'save_dir': 'deepsrshxc_checkpoints',
        'save_interval': 10,
        'log_interval': 5,
        'num_workers': 4,
        'val_ratio': 0.2  # 验证集比例
    }
    
    # 数据路径 - 所有mol2文件在同一个文件夹
    mol2_directory = "/Users/jiaoyuan/Documents/GitHub/ADOPTXC/dataste_mol"  # 修改为你的mol2文件夹路径
    
    # 创建数据可视化
    viz_dir = 'deepsrshxc_checkpoints/visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    plot_molecule_distribution(mol2_directory, os.path.join(viz_dir, 'molecule_distribution.png'))
    
    # 创建数据加载器
    try:
        train_loader, val_loader = create_data_loaders(
            mol2_directory, 
            batch_size=config['batch_size'],
            val_ratio=config['val_ratio'],
            num_workers=config['num_workers']
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return
    
    # 初始化模型
    model = DeepsRSHXC(
        num_heads=8,
        num_gat_layers=2,
        dropout=0.6
    )
    
    # 创建训练器并开始训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()

def resume_training(checkpoint_path, config_updates=None):
    """从检查点恢复训练"""
    checkpoint = torch.load(checkpoint_path)
    
    # 更新配置
    config = checkpoint['config']
    if config_updates:
        config.update(config_updates)
    
    # 数据路径
    mol2_directory = "/Users/jiaoyuan/Documents/GitHub/ADOPTXC/dataste_mol"  # 修改为你的mol2文件夹路径
    
    # 重新创建数据加载器
    train_loader, val_loader = create_data_loaders(
        mol2_directory, 
        batch_size=config['batch_size'],
        val_ratio=config.get('val_ratio', 0.2),
        num_workers=config.get('num_workers', 4)
    )
    
    # 初始化模型
    model = DeepsRSHXC(
        num_heads=8,
        num_gat_layers=2,
        dropout=0.6
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # 加载检查点
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    trainer.train_history = checkpoint['train_history']
    
    # 继续训练
    trainer.train()

if __name__ == "__main__":
    # 正常训练
    main()
    
    # 或者从检查点恢复训练
    # resume_training("deepsrshxc_checkpoints/best_model.pth")
