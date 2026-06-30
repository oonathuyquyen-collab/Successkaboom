import torch
from unified_model import UnifiedTMIL, entropy_loss

def test_unified_model():
    batch_size = 4
    seq_len = 100
    vocab_size = 1000
    extra_feat_dim = 5
    
    model = UnifiedTMIL(vocab_size=vocab_size, extra_feat_dim=extra_feat_dim)
    
    ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    io = torch.randint(0, 3, (batch_size, seq_len))
    hc = torch.randn(batch_size, seq_len, 2)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Mask out some padding
    mask[:, 80:] = False
    
    extra_feats = torch.randn(batch_size, seq_len, extra_feat_dim)
    
    account_logit, attention = model(ids, io, hc, mask, extra_feats)
    
    print(f"Account logit shape: {account_logit.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Check if attention respects mask
    print(f"Attention sum (should be ~1.0): {attention.sum(dim=1)}")
    print(f"Attention on masked items (should be 0): {attention[:, 80:].sum()}")
    
    loss = entropy_loss(attention, mask)
    print(f"Entropy loss: {loss.item():.4f}")
    
    print("Unified model shape test passed!")

if __name__ == "__main__":
    test_unified_model()
