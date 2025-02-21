from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    num_hidden_layers: int = 12
    num_attention_heads: int = 9
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    num_experts: int = 8
    batch_size: int = 1
    max_length: int = 128
    learning_rate: float = 1e-4
    max_steps: int = 10000
    checkpoint_steps: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    dropout_prob: float = 0.1
    use_alibi: bool = True
    use_rope: bool = True
    num_latents: int = 64  # Number of latent queries
    
    def __post_init__(self):
        # Validate configuration
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads" 