"""
Advanced Diffusion Model Configuration
This file contains all the advanced techniques and hyperparameters for improved diffusion training
"""

class AdvancedDiffusionConfig:
    # Model Architecture
    MODEL_CONFIG = {
        'dim': 128,                    # Base dimension
        'init_dim': 128,               # Initial dimension
        'out_dim': 2,                  # Output dimension (stereo)
        'channels': 2,                 # Input channels
        'self_condition': True,        # Self-conditioning
        'resnet_block_groups': 8,      # ResNet block groups
        'learned_variance': False,     # Learned variance
        'use_enhanced_attention': True, # Enhanced attention
        'use_multi_scale_loss': True,  # Multi-scale loss
    }
    
    # Diffusion Parameters
    DIFFUSION_CONFIG = {
        'timesteps': 1000,             # Number of diffusion steps
        'sampling_timesteps': 25,      # DDIM sampling steps
        'loss_type': 'l1',             # Loss type (l1 or l2)
        'objective': 'pred_noise',     # Prediction objective
        'beta_schedule': 'improved_cosine', # Improved cosine schedule
        'ddim_sampling_eta': 0,        # DDIM eta
        'auto_normalize': False,       # Auto normalization
        'min_snr_loss_weight': False,  # Min SNR loss weight
    }
    
    # Classifier-Free Guidance
    CFG_CONFIG = {
        'cfg_scale': 7.5,             # CFG scale
        'use_cfg': True,              # Enable CFG
        'uncond_dropout': 0.1,        # Unconditional dropout
        'cfg_training': True,          # Enable CFG during training
    }
    
    # Training Parameters
    TRAINING_CONFIG = {
        'batch_size': 8,               # Batch size
        'learning_rate': 1e-4,        # Learning rate
        'weight_decay': 1e-6,         # Weight decay
        'gradient_clip': 1.0,         # Gradient clipping
        'ema_decay': 0.9999,          # EMA decay
        'warmup_steps': 1000,         # Warmup steps
        'scheduler_t0': 1000,         # Scheduler T0
        'scheduler_t_mult': 2,        # Scheduler T multiplier
        'scheduler_eta_min': 1e-6,    # Scheduler minimum LR
    }
    
    # Multi-Scale Loss
    MULTI_SCALE_CONFIG = {
        'scales': [1, 2, 4],          # Multi-scale factors
        'weights': [1.0, 0.5, 0.25],  # Scale weights
    }
    
    # Attention Enhancement
    ATTENTION_CONFIG = {
        'spatial_temporal_heads': 8,   # Spatial-temporal attention heads
        'spatial_temporal_dim_head': 64, # Spatial-temporal attention dim
        'cross_modal_heads': 8,        # Cross-modal attention heads
        'cross_modal_dim_head': 64,    # Cross-modal attention dim
        'dropout': 0.1,               # Attention dropout
    }
    
    # Noise Schedule
    NOISE_SCHEDULE_CONFIG = {
        'num_timesteps': 1000,        # Number of timesteps
        'beta_start': 1e-4,           # Beta start
        'beta_end': 0.02,             # Beta end
        'cosine_offset': 0.008,       # Cosine offset
        'cosine_scale': 1.008,        # Cosine scale
        'use_improved_schedule': True, # Use improved schedule
    }
    
    # Sampling Parameters
    SAMPLING_CONFIG = {
        'use_cfg': True,              # Use CFG during sampling
        'cfg_scale': 7.5,             # CFG scale for sampling
        'ddim_steps': 25,             # DDIM sampling steps
        'eta': 0.0,                   # DDIM eta
        'silence_mask_sampling': True, # Silence mask sampling
        'dynamic_threshold': False,    # Dynamic thresholding
        'dynamic_threshold_percentile': 0.95, # Dynamic threshold percentile
    }
    
    # Evaluation Parameters
    EVAL_CONFIG = {
        'eval_epoch': 5,              # Evaluation frequency
        'save_freq': 10,              # Save frequency
        'vis_freq': 5,                # Visualization frequency
        'metrics': ['l2_distance', 'mel_l2'], # Evaluation metrics
    }
    
    # Advanced Techniques
    ADVANCED_TECHNIQUES = {
        'use_ema': True,              # Use Exponential Moving Average
        'use_improved_schedule': True, # Use improved noise schedule
        'use_cfg_training': True,     # Use CFG during training
        'use_cfg_sampling': True,     # Use CFG during sampling
        'use_multi_scale_loss': True, # Use multi-scale loss
        'use_enhanced_attention': True, # Use enhanced attention
        'use_weighted_loss': True,    # Use weighted loss
    }
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration"""
        return cls.MODEL_CONFIG
    
    @classmethod
    def get_diffusion_config(cls):
        """Get diffusion configuration"""
        return cls.DIFFUSION_CONFIG
    
    @classmethod
    def get_cfg_config(cls):
        """Get CFG configuration"""
        return cls.CFG_CONFIG
    
    @classmethod
    def get_training_config(cls):
        """Get training configuration"""
        return cls.TRAINING_CONFIG
    
    @classmethod
    def get_multi_scale_config(cls):
        """Get multi-scale configuration"""
        return cls.MULTI_SCALE_CONFIG
    
    @classmethod
    def get_attention_config(cls):
        """Get attention configuration"""
        return cls.ATTENTION_CONFIG
    
    @classmethod
    def get_noise_schedule_config(cls):
        """Get noise schedule configuration"""
        return cls.NOISE_SCHEDULE_CONFIG
    
    @classmethod
    def get_sampling_config(cls):
        """Get sampling configuration"""
        return cls.SAMPLING_CONFIG
    
    @classmethod
    def get_eval_config(cls):
        """Get evaluation configuration"""
        return cls.EVAL_CONFIG
    
    @classmethod
    def get_advanced_techniques_config(cls):
        """Get advanced techniques configuration"""
        return cls.ADVANCED_TECHNIQUES 