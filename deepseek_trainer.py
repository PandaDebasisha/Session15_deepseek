import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
import os
from typing import Optional
import logging
from deepseek_model import DeepSeekModel
from model_config import DeepSeekConfig
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class DeepSeekTrainer:
    def __init__(
        self,
        model: DeepSeekModel,
        tokenizer: AutoTokenizer,
        config: DeepSeekConfig,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        load_checkpoint: Optional[str] = None
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.scaler = GradScaler()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize or load from checkpoint
        self.start_step = 0
        if load_checkpoint:
            self.start_step = self.load_checkpoint(load_checkpoint)
        else:
            self.start_step = self.find_latest_checkpoint()
            
    def find_latest_checkpoint(self):
        """Find and load the latest checkpoint if it exists."""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"))
        if not checkpoint_files:
            logging.info("No existing checkpoints found. Starting from step 0.")
            return 0
            
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        step = self.load_checkpoint(latest_checkpoint)
        logging.info(f"Resuming from checkpoint at step {step}")
        return step
            
    def save_checkpoint(self, step: int, optimizer: torch.optim.Optimizer):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'step': step,
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint at step {step}")
        
        # Remove older checkpoints, keep only the last 3
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt")),
                                key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for old_checkpoint in checkpoint_files[:-3]:
            os.remove(old_checkpoint)
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer_state = checkpoint['optimizer_state_dict']
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            step = checkpoint['step']
            logging.info(f"Successfully loaded checkpoint from step {step}")
            return step
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return 0
        
    def train(self, train_dataloader: DataLoader):
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if hasattr(self, 'optimizer_state'):
            optimizer.load_state_dict(self.optimizer_state)
        
        step = self.start_step
        while step < self.config.max_steps:
            try:
                for batch in train_dataloader:
                    # Ensure proper tensor dimensions and types
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Ensure input_ids are properly shaped
                    batch_size, seq_length = input_ids.size()
                    if seq_length > self.config.max_length:
                        input_ids = input_ids[:, :self.config.max_length]
                        attention_mask = attention_mask[:, :self.config.max_length]
                    
                    # Create labels (shifted input_ids)
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        # Forward pass
                        lm_logits, load_balancing_loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        # Calculate loss
                        shift_logits = lm_logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        loss = nn.CrossEntropyLoss()(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Add load balancing loss
                        total_loss = loss + 0.01 * load_balancing_loss

                    # Backward pass
                    self.scaler.scale(total_loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    # Log metrics
                    if step % 10 == 0:
                        logging.info(
                            f"Step {step} - Loss: {loss.item():.4f}, "
                            f"Load Balancing Loss: {load_balancing_loss.item():.4f}"
                        )
                    
                    # Save checkpoint and generate sample
                    if step % self.config.checkpoint_steps == 0:
                        self.save_checkpoint(step, optimizer)
                        self.generate_sample(input_ids[0])
                        
                    step += 1
                    if step >= self.config.max_steps:
                        break
                        
            except Exception as e:
                logging.error(f"Error during training at step {step}: {str(e)}")
                raise  # Re-raise the exception for debugging
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'final_step': step
        }, final_path)
        logging.info("Training completed. Final model saved.")
        
    def generate_sample(self, input_ids):
        self.model.eval()
        with torch.no_grad():
            # Ensure input_ids are properly shaped
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            # Generate text
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.config.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Sample generation:\n{generated_text}")
        self.model.train()

def main():
    # Initialize config
    config = DeepSeekConfig(
        batch_size=1,
        max_steps=10001,
        checkpoint_steps=100,
    )
    
    # Load dataset
    datasets = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        streaming=True,
        split="train"
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Update config
    config.vocab_size = len(tokenizer)
    
    # Initialize model
    model = DeepSeekModel(config)
    
    # Create dataloader
    def collate_fn(examples):
        return tokenizer(
            [ex["text"] for ex in examples],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
    
    train_dataloader = DataLoader(
        datasets,
        batch_size=config.batch_size,
        collate_fn=collate_fn
    )
    
    # Initialize and run trainer
    trainer = DeepSeekTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir="deepseek_checkpoints"
    )
    
    trainer.train(train_dataloader)

if __name__ == "__main__":
    main() 