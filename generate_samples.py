import torch
import logging
from transformers import AutoTokenizer
from deepseek_model import DeepSeekModel
from model_config import DeepSeekConfig
import os
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler()
    ]
)

class DeepSeekGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        
        # Load the saved model and config
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Initialize model
        self.model = DeepSeekModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Cache special tokens
        self.special_tokens = {
            'pad': self.tokenizer.pad_token_id,
            'eos': self.tokenizer.eos_token_id,
            'bos': self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
        }
        
        logging.info(f"Model loaded successfully from {model_path}")
        
    @torch.inference_mode()
    def generate(self, prompt: str, 
                max_length: int = 60,
                min_length: int = 20,
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = 1.5,
                no_repeat_ngram_size: int = 3,
                presence_penalty: float = 0.5,
                frequency_penalty: float = 0.5):
        """Improved text generation with enhanced repetition controls"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Track token usage with more sophisticated counting
        token_frequencies = torch.zeros(self.config.vocab_size, device=self.device)
        token_presence = torch.zeros(self.config.vocab_size, device=self.device)
        generated_ngrams = {}
        last_n_tokens = []
        
        # Initialize presence counts from prompt
        for token in input_ids[0]:
            token_presence[token] = 1
        
        for _ in range(max_length - input_ids.shape[1]):
            outputs, _ = self.model(input_ids, attention_mask)
            next_token_logits = outputs[:, -1, :].clone()
            
            # Apply presence penalty (for tokens that appeared at all)
            next_token_logits -= presence_penalty * token_presence
            
            # Apply frequency penalty (scaled by how often tokens appeared)
            if token_frequencies.max() > 0:
                scaled_frequencies = token_frequencies / token_frequencies.max()
                next_token_logits -= frequency_penalty * scaled_frequencies
            
            # Apply stronger repetition penalty for recent tokens
            if last_n_tokens:
                recent_tokens = set(last_n_tokens[-5:])  # Last 5 tokens get extra penalty
                for token_id in recent_tokens:
                    next_token_logits[0, token_id] /= repetition_penalty * 1.5
            
            # Prevent n-gram repetition
            if no_repeat_ngram_size > 0 and len(last_n_tokens) >= no_repeat_ngram_size:
                ngram = tuple(last_n_tokens[-(no_repeat_ngram_size-1):])
                for token_id in range(self.config.vocab_size):
                    if ngram + (token_id,) in generated_ngrams:
                        next_token_logits[0, token_id] = float('-inf')
            
            # Prevent special tokens and very common words from being overused
            for token_id in self.special_tokens.values():
                if token_id is not None:
                    next_token_logits[0, token_id] = float('-inf')
            
            # Apply dynamic temperature based on token frequencies
            local_temp = temperature * (1 + 0.1 * (token_frequencies.max() > 2).float())
            next_token_logits /= local_temp
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token with enhanced diversity
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update tracking
            token_frequencies[next_token] += 1
            token_presence[next_token] = 1
            last_n_tokens.append(next_token.item())
            if len(last_n_tokens) > no_repeat_ngram_size:
                last_n_tokens.pop(0)
            
            if no_repeat_ngram_size > 0 and len(last_n_tokens) >= no_repeat_ngram_size:
                generated_ngrams[tuple(last_n_tokens)] = True
            
            # Update input tensors
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
            
            # Early stopping check
            if input_ids.size(1) >= min_length and next_token.item() == self.special_tokens['eos']:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    # Path to your saved model
    model_path = "deepseek_checkpoints/final_model.pt"
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return
        
    # Initialize generator
    generator = DeepSeekGenerator(model_path)
    
    # Sample prompts
    prompts = [
        "Neutrino Masses and OscillationsView this Special Issue Challenges in Double Beta Decaye",
        "Music Festivals Across America",
        "In the depths of space",
        "The most interesting scientific discovery",
        "When I think about consciousness"
    ]
    
    # Generate samples with different parameters for each prompt
    logging.info("Generating samples...")
    for i, prompt in enumerate(prompts, 1):
        logging.info(f"\nSample {i}:")
        logging.info(f"Prompt: {prompt}")
        try:
            temperature = 0.9 + (i * 0.15)  # Higher base temperature: 1.05 to 1.65
            generated_text = generator.generate(
                prompt,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                presence_penalty=0.5,
                frequency_penalty=0.5,
                min_length=20,
                max_length=60
            )
            logging.info(f"Generated text (temp={temperature:.1f}):\n{generated_text}\n")
            logging.info("-" * 50)
        except Exception as e:
            logging.error(f"Error generating sample {i}: {str(e)}")

if __name__ == "__main__":
    main() 