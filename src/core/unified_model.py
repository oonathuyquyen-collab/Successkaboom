import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedTMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=0, use_tcn=True, direction_mode="io_embed"):
        super().__init__()
        self.direction_mode = direction_mode
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn: 
            self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Gated Attention now takes h_k concatenated with extra engineered features
        attn_input_dim = embed_dim + extra_feat_dim
        self.attn_V = nn.Linear(attn_input_dim, 128)
        self.attn_U = nn.Linear(attn_input_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ids, io, hc, mask, extra_feats=None):
        h = self.cp_embed(ids) + self.hc_proj(hc)
        if self.direction_mode == "io_embed": 
            h = h + self.io_embed(io)
        h = self.norm(h)
        
        if self.use_tcn: 
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
            
        # Prepare input for attention
        attn_input = h
        if extra_feats is not None:
            # extra_feats shape: (batch_size, seq_len, extra_feat_dim)
            attn_input = torch.cat([h, extra_feats], dim=-1)
            
        s = self.attn_w(torch.tanh(self.attn_V(attn_input)) * torch.sigmoid(self.attn_U(attn_input))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        
        if self.direction_mode == "hardmask":
            outb = (io == 1)
            has = outb.any(dim=1, keepdim=True)
            s = s.masked_fill(has & ~outb & mask, -1e9)
            
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        
        # Account-level classification score
        account_logit = self.classifier(z).squeeze(-1)
        
        # Transaction-level ranking score is directly the attention weights 'a'
        return account_logit, a

def entropy_loss(attention_weights, mask):
    # attention_weights shape: (batch_size, seq_len)
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=1)
    # Average over batch
    return entropy.mean()
