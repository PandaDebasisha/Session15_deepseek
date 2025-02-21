import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_config import DeepSeekConfig

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # Handle position_ids
    if position_ids is not None:
        # Reshape position_ids to match the expected dimensions
        position_ids = position_ids.view(-1)  # Flatten position_ids
        cos = cos[:, :, position_ids]  # Use advanced indexing instead of index_select
        sin = sin[:, :, position_ids]
        
        # Reshape cos and sin back to match q and k dimensions
        cos = cos.view(1, 1, q.size(2), q.size(3))
        sin = sin.view(1, 1, q.size(2), q.size(3))
    
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Match latent dimension to input sequence length
        self.num_latents = config.max_length  # Use max_length instead of fixed number
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_latents, self.hidden_size) / self.hidden_size ** 0.5)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project inputs
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape and transpose
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        cos, sin = self.rotary_emb(q, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Process latent queries - match sequence length
        latents = self.latent_queries[:, :seq_length, :].expand(batch_size, -1, -1)
        latent_q = self.q_proj(latents).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(latent_q, k.transpose(-2, -1)) * self.scaling
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
        # Compute attention weights and output
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        
        return self.o_proj(attn_output)

class GLUFeedForward(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        hidden = F.gelu(x1) * x2
        return self.dropout(self.w2(hidden))

class MoELayer(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        
        self.experts = nn.ModuleList([
            GLUFeedForward(config) for _ in range(self.num_experts)
        ])
        
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Get routing probabilities
        route_logits = self.router(x)
        routing_weights = F.softmax(route_logits, dim=-1)
        
        # Compute load balancing loss
        mean_routing = routing_weights.mean(dim=[0, 1])
        target_routing = torch.ones_like(mean_routing) / self.num_experts
        load_balancing_loss = F.kl_div(
            F.log_softmax(mean_routing, dim=-1),
            target_routing,
            reduction='sum'
        )
        
        # Dispatch to experts
        combined_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_weights = routing_weights[..., i:i+1]
            expert_output = expert(x)
            combined_output += expert_output * expert_weights
            
        return self.dropout(combined_output), load_balancing_loss

class TransformerBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config)
        self.moe = MoELayer(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, attention_mask=None, position_ids=None):
        # Pre-norm
        attn_output = self.attention(self.ln1(x), attention_mask, position_ids)
        x = x + attn_output
        
        # MoE with pre-norm
        hidden_states, load_balancing_loss = self.moe(self.ln2(x))
        x = x + hidden_states
        
        return x, load_balancing_loss

class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Get position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Process through transformer layers
        total_load_balancing_loss = 0
        for layer in self.layers:
            hidden_states, load_balancing_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            total_load_balancing_loss += load_balancing_loss
            
        # Final normalization and output
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, total_load_balancing_loss

    def generate(self, input_ids, max_length, **kwargs):
        self.eval()
        with torch.no_grad():
            cur_len = input_ids.size(1)
            
            while cur_len < max_length:
                outputs = self(input_ids)
                next_token_logits = outputs[0][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                cur_len += 1
                
                if next_token.item() == kwargs.get('pad_token_id', None) or \
                   next_token.item() == kwargs.get('eos_token_id', None):
                    break
                    
            return input_ids 