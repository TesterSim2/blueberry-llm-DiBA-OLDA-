import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 1.58-Bit Quantization Kernels
# ==========================================
class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, flg_bit=True):
        super().__init__(in_features, out_features, bias=bias)
        self.flg_bit = flg_bit

    def activation_quant(self, x):
        """Per-token 8-bit quantization with STE."""
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y + (x - x.detach())

    def weight_quant(self, w):
        """1.58-bit {-1, 0, 1} weight quantization with STE."""
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w_scaled = w * scale
        w_quant = w_scaled.round().clamp_(-1, 1) / scale
        return w_quant + (w - w.detach())

    def forward(self, x):
        if not self.flg_bit:
            return F.linear(x, self.weight, self.bias)

        x_q = self.activation_quant(x)
        w_q = self.weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

# ==========================================
# 2. Neuro-Symbolic Gate
# ==========================================
class NeuroSymbolicGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Lightweight temporal context window
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.proj = nn.Linear(d_model, 1)
        self.act = nn.Sigmoid()

        # Bias initialization to start near 0.8 (Lambda=0.8)
        nn.init.constant_(self.proj.bias, 1.4)

    def forward(self, x):
        # x: (B, S, D) -> Permute for Conv1d -> (B, D, S)
        x_perm = x.permute(0, 2, 1)
        feats = self.conv(x_perm).permute(0, 2, 1)
        return self.act(self.proj(feats))

# ==========================================
# 3. DiBA-OLDA Attention Layer (The Replacement)
# ==========================================
class Gated_DiBA_OLDA_Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.dim
        self.n_heads = args.n_heads
        self.d_head = args.head_dim
        self.d_latent = args.kv_lora_rank  # Blueberry likely uses this name for c_KV
        self.use_bitnet = getattr(args, 'use_bitnet', True)

        # 1. Down-Projection (Compression) - High Precision
        self.W_Down = nn.Linear(self.d_model, self.d_latent, bias=False)

        # 2. Orthogonal Up-Projections (Signal & Noise) - 1.58-bit
        # Output dim = n_heads * d_head
        self.W_U_Sig = BitLinear(self.d_latent, self.n_heads * self.d_head, bias=False, flg_bit=self.use_bitnet)
        self.W_U_Noise = BitLinear(self.d_latent, self.n_heads * self.d_head, bias=False, flg_bit=self.use_bitnet)

        # 3. Query Projections (Split Q) - 1.58-bit
        self.W_Q_Sig = BitLinear(self.d_model, self.n_heads * self.d_head, bias=False, flg_bit=self.use_bitnet)
        self.W_Q_Noise = BitLinear(self.d_model, self.n_heads * self.d_head, bias=False, flg_bit=self.use_bitnet)

        # 4. Value & Output - 1.58-bit
        self.W_U_Val = BitLinear(self.d_latent, self.n_heads * self.d_head, bias=False, flg_bit=self.use_bitnet)
        self.W_O = BitLinear(self.n_heads * self.d_head, self.d_model, bias=False, flg_bit=self.use_bitnet)

        # 5. Gating
        self.gate = NeuroSymbolicGate(self.d_model)

        # 6. Orthogonal Initialization
        nn.init.orthogonal_(self.W_U_Sig.weight)
        nn.init.orthogonal_(self.W_U_Noise.weight)

    def get_ortho_loss(self):
        # Calculate orthogonality penalty on shadow weights
        w1 = self.W_U_Sig.weight.T
        w2 = self.W_U_Noise.weight.T
        interaction = torch.matmul(w1.T, w2)
        # Normalize by size to keep loss scale manageable
        return (torch.norm(interaction, p='fro') ** 2) / (w1.shape[0] * w1.shape[1])

    def forward(self, x, freqs_cis):
        B, S, _ = x.shape

        # A. Gate
        lambda_g = self.gate(x).unsqueeze(1)  # (B, 1, S, 1)

        # B. Compression
        c_kv = self.W_Down(x)

        # C. Projections (Quantized)
        k_sig = self.W_U_Sig(c_kv).view(B, S, self.n_heads, self.d_head)
        k_noise = self.W_U_Noise(c_kv).view(B, S, self.n_heads, self.d_head)
        q_sig = self.W_Q_Sig(x).view(B, S, self.n_heads, self.d_head)
        q_noise = self.W_Q_Noise(x).view(B, S, self.n_heads, self.d_head)
        v = self.W_U_Val(c_kv).view(B, S, self.n_heads, self.d_head)

        # D. RoPE (Signal Only)
        # Assuming BlueberryLLM uses the same apply_rope signature
        q_sig, k_sig = apply_rope(q_sig, k_sig, freqs_cis)

        # Transpose to (B, H, S, D)
        q_sig, k_sig = q_sig.transpose(1, 2), k_sig.transpose(1, 2)
        q_noise, k_noise = q_noise.transpose(1, 2), k_noise.transpose(1, 2)
        v = v.transpose(1, 2)

        # E. Attention Scores
        scale = 1.0 / math.sqrt(self.d_head)
        # Simple causal mask (assuming flash attention isn't available for custom diff logic yet)
        mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)

        s_sig = (torch.matmul(q_sig, k_sig.transpose(-2, -1)) * scale) + mask
        s_noise = (torch.matmul(q_noise, k_noise.transpose(-2, -1)) * scale) + mask

        a_sig = F.softmax(s_sig, dim=-1)
        a_noise = F.softmax(s_noise, dim=-1)

        # F. Differential Subtraction
        a_diff = a_sig - (lambda_g * a_noise)

        # G. Output
        out = torch.matmul(a_diff, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.W_O(out)
