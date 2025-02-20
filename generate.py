import torch
from model import TransformerLM
from train import SimpleTokenizer

def generate_text(model, tokenizer, seed_text="Once upon a", max_length=200, temperature=0.8, device='cpu'):
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

if __name__ == "__main__":
    # Load the saved model
    checkpoint = torch.load('poe_lm_model.pt', weights_only=False)
    
    # Extract model dimensions from checkpoint to ensure compatibility
    vocab_size = len(checkpoint['tokenizer'].char_to_idx)
    
    # Create model with the same dimensions as during training
    loaded_model = TransformerLM(
        vocab_size=vocab_size,
        d_model=384,  # Match the dimension from training
        nhead=6,      # Match the head count from training
        num_layers=6  # Match the layer count from training
    )
    
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_tokenizer = checkpoint['tokenizer']
    
    # Set device - support for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    loaded_model.to(device)

    # Generate text
    generated_text = generate_text(
        model=loaded_model,
        tokenizer=loaded_tokenizer,
        seed_text="Once upon a",
        temperature=0.7,
        device=device
    )
    print(generated_text)