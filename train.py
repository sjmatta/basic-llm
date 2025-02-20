import torch
from torch.utils.data import Dataset, DataLoader
import math
import os
from model import TransformerLM

# Simple tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        
        for i, char in enumerate(sorted(unique_chars)):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        for text in texts:
            self.data.extend(self.tokenizer.encode(text))
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(x), torch.tensor(y)

# Generate text function for evaluation
def generate_sample(model, tokenizer, seed_text="Once upon a", max_length=100, temperature=0.7, device='cpu'):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(seed_text)).unsqueeze(0).to(device)
    generated = list(input_ids[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
                
    return tokenizer.decode(generated)

# Training function with checkpoints
def train_model(model, dataloader, epochs=100, lr=0.001, device='cpu', checkpoint_dir='checkpoints', 
                checkpoint_interval=5, seed_text="Once upon a midnight dreary, "):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate average loss for this epoch        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint at regular intervals and if loss improves
        if (epoch + 1) % checkpoint_interval == 0 or avg_loss < best_loss:
            checkpoint_path = os.path.join(checkpoint_dir, f'poe_model_epoch_{epoch+1}.pt')
            
            # Save model and tokenizer
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'tokenizer': tokenizer
            }, checkpoint_path)
            
            print(f"Checkpoint saved at {checkpoint_path}")
            
            # Generate and display a sample
            sample_text = generate_sample(model, tokenizer, seed_text=seed_text, device=device)
            print(f"\nSample generation at epoch {epoch+1}:")
            print(f"{sample_text}\n")
            
            # Update best loss if improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(checkpoint_dir, 'poe_model_best.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'tokenizer': tokenizer
                }, best_model_path)
                print(f"New best model saved with loss: {avg_loss:.4f}")

# Main training script
if __name__ == "__main__":
    # Load the Poe corpus
    with open('poe_complete_works.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Create a smaller subset for faster training if needed
    # Adjust this to use more of the corpus for better results
    train_text = text[:1000000]  # First 1M characters - expanded from 500K

    # Initialize and fit tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit([train_text])
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset
    dataset = TextDataset([train_text], tokenizer, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Increased batch size

    # Initialize model
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        nhead=6,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.2
    )

    # Check for MPS (Apple Silicon) or fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Train the model with checkpoints
    train_model(
        model=model,
        dataloader=dataloader,
        epochs=100,             # Increased epochs
        lr=0.0003,
        device=device,
        checkpoint_interval=5,  # Save every 5 epochs
        seed_text="Once upon a midnight dreary, "
    )