import math
import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence modeling in network traffic."""

    def __init__(self, d_model, max_len=512, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _extend(self, new_len: int):
        if new_len > self.pe.size(1):
            self.pe = self._build_pe(new_len, self.d_model).to(self.pe.device, self.pe.dtype)

    def forward(self, x):
        """
        Args:
            x: tensor (B, L, d_model)
        """
        L = x.size(1)
        if L > self.pe.size(1):
            self._extend(L)

        pe = self.pe[:, :L].to(dtype=x.dtype, device=x.device)
        x = x + pe
        return self.dropout(x)


class ExplainableTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with native support for attention weight extraction.
    
    In standard training (return_attention=False), it runs optimized attention.
    When return_attention=True, it returns (output, attn_weights) with shape (B, h, L, L),
    enabling intrinsic XAI, attention heatmaps, and gradient-weighted attention attribution.
    """

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, src, src_key_padding_mask=None, return_attention=False):
        kwargs = dict(key_padding_mask=src_key_padding_mask, need_weights=return_attention)
        if return_attention:
            kwargs['average_attn_weights'] = False

        attn_out, attn_weights = self.self_attn(src, src, src, **kwargs)
        src = self.norm1(src + self.dropout1(attn_out))
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ffn_out))

        if return_attention:
            return src, attn_weights
        return src


class BiflowTransformer(nn.Module):
    """Explainability-Ready Sequence Transformer for IoMT network intrusion detection.
    
    Supports:
      - Variable sequence lengths (num_pkts in 10, 20, 50, 100).
      - Configurable hyperparameters (nlayers, nhead, d_model, dim_feedforward).
      - Robust padding mask detection (checking all fields zero).
      - Intrinsic XAI: extraction of attention matrices across all layers and heads.
    """

    def __init__(self, num_pkts, num_fields, num_classes=10, dropout=0.1, **kwargs):
        super().__init__()

        self.num_pkts = num_pkts
        self.num_fields = num_fields
        self.d_model = kwargs.get('d_model', 128)
        self.nhead = kwargs.get('nhead', kwargs.get('nheads', 4))
        self.dim_feedforward = kwargs.get('dim_feedforward', 512)
        self.num_layers = kwargs.get('nlayers', kwargs.get('num_layers', 2))
        self.out_features_size = kwargs.get('out_features_size', 200)
        if self.out_features_size is None:
            self.out_features_size = 200

        self.in_proj = nn.Linear(num_fields, self.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=num_pkts + 1, dropout=dropout)

        self.layers = nn.ModuleList([
            ExplainableTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(self.num_layers)
        ])

        self.fc1 = nn.Linear(self.d_model, self.out_features_size)
        self.fc = nn.Linear(self.out_features_size, num_classes)
        self.head_var = 'fc'

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        if return_attention:
            feats, attns = self.extract_features(x, return_attention=True)
            out = self.fc(F.gelu(feats))
            return out, attns
        out = F.gelu(self.extract_features(x))
        out = self.fc(out)
        return out

    def extract_features(self, x, return_attention=False):
        # Expected input x shape: (B, 1, num_pkts, num_fields) or (B, num_pkts, num_fields)
        if x.dim() == 4:
            x = x.squeeze(1)
        B, L, F_dim = x.shape

        # Robust padding mask: a packet is padding if all fields are 0
        pad_mask = torch.all(x == 0, dim=-1)  # (B, L) boolean
        # Prepend False for CLS token (never masked)
        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1
        )  # (B, L+1)

        # Linear projection to d_model
        x = self.in_proj(x)  # (B, L, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, L+1, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through encoder layers
        attentions = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, src_key_padding_mask=pad_mask, return_attention=True)
                attentions.append(attn)
            else:
                x = layer(x, src_key_padding_mask=pad_mask, return_attention=False)

        cls_output = x[:, 0]  # CLS token representation (B, d_model)
        features = self.fc1(cls_output)  # (B, out_features_size)

        if return_attention:
            return features, attentions
        return features

    def get_attention_maps(self, x):
        """Convenience method for XAI analysis: returns (predictions, attention_weights_per_layer)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_attention=True)


class MedDefenderTransformer(nn.Module):
    """Dual-Stream Spatial-Temporal Transformer inspired by the MedDefender-MHAN framework (Alqazzaz, PLoS One 2026).
    
    Combines:
      - 1D-CNN stream for local spatial burst signatures.
      - Transformer stream for long-range temporal dependencies.
      - Feature Fusion + Multi-Head Attention pooling.
      - Native explainability preservation via attention matrices.
    """

    def __init__(self, num_pkts, num_fields, num_classes=10, dropout=0.1, **kwargs):
        super().__init__()

        self.num_pkts = num_pkts
        self.num_fields = num_fields
        self.nlayers = int(kwargs.get('nlayers', 1))
        if self.nlayers < 1:
            raise ValueError(f"nlayers must be >= 1, got {self.nlayers}")
        self.d_model = kwargs.get('d_model', 128)
        self.nhead = kwargs.get('nhead', kwargs.get('nheads', 4))
        self.out_features_size = kwargs.get('out_features_size', 200)
        if self.out_features_size is None:
            self.out_features_size = 200

        # Stream 1: 1D-CNN for local burst dynamics
        # Input: (B, num_fields, num_pkts)
        self.cnn_stream = nn.Sequential(
            nn.Conv1d(num_fields, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )

        # Stream 2: Transformer Stream for long-range temporal dependencies
        self.in_proj = nn.Linear(num_fields, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=num_pkts, dropout=dropout)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        if self.nlayers == 1:
            # Keep single-layer attribute name so 1-layer trained checkpoints stay strict=True compatible
            self.transformer_layer = ExplainableTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
        else:
            self.transformer_layers = nn.ModuleList([
                ExplainableTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu'
                )
                for _ in range(self.nlayers)
            ])

        # Feature Fusion Layer (concatenation along channel dimension + projection)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-Head Attention Encoder over fused features
        self.mha = nn.MultiheadAttention(self.d_model, self.nhead, dropout=dropout, batch_first=True)

        # Classification Head
        self.fc1 = nn.Linear(self.d_model, self.out_features_size)
        self.fc = nn.Linear(self.out_features_size, num_classes)
        self.head_var = 'fc'

    def forward(self, x, return_attention=False):
        if return_attention:
            feats, attn = self.extract_features(x, return_attention=True)
            out = self.fc(F.gelu(feats))
            return out, attn
        out = F.gelu(self.extract_features(x))
        out = self.fc(out)
        return out

    def extract_features(self, x, return_attention=False):
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, T, F)
        B, T, F_dim = x.shape

        # CNN Stream: (B, F, T) -> (B, d_model, T) -> (B, T, d_model)
        x_cnn = x.transpose(1, 2)
        f_cnn = self.cnn_stream(x_cnn).transpose(1, 2)

        # Transformer Stream: (B, T, F) -> (B, T, d_model)
        x_trans = self.in_proj(x)
        x_trans = self.positional_encoding(x_trans)
        pad_mask = torch.all(x == 0, dim=-1)

        if self.nlayers == 1:
            f_trans = self.transformer_layer(x_trans, src_key_padding_mask=pad_mask, return_attention=False)
        else:
            f_trans = x_trans
            for layer in self.transformer_layers:
                f_trans = layer(f_trans, src_key_padding_mask=pad_mask, return_attention=False)

        # Fusion: concatenate along feature dimension -> (B, T, d_model)
        f_fused = self.fusion(torch.cat([f_cnn, f_trans], dim=-1))

        # Final Multi-Head Attention Pooling
        mha_kwargs = dict(key_padding_mask=pad_mask, need_weights=return_attention)
        if return_attention:
            mha_kwargs['average_attn_weights'] = False

        attn_out, mha_weights = self.mha(
            f_fused, f_fused, f_fused,
            **mha_kwargs
        )

        # Masked Average Pooling across time steps (ignoring padding)
        keep = (~pad_mask).unsqueeze(-1).float()  # (B, T, 1)
        f_pooled = (attn_out * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # (B, d_model)
        features = self.fc1(f_pooled)  # (B, out_features_size)

        if return_attention:
            return features, mha_weights
        return features

    def get_attention_maps(self, x):
        """Convenience method for XAI analysis: returns (predictions, attention_weights)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_attention=True)
