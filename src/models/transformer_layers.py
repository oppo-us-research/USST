""" Code modified from: https://github.com/stevenlsw/hoi-forecast/tree/master/networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def get_pad_mask(seq, pad_idx=0):
    if not isinstance(pad_idx, int):
        raise TypeError("<pad_index> has to be an int!")
    if seq.dim() == 3:
        return seq != pad_idx  # (B, T, T)
    return (seq != pad_idx).unsqueeze(1) # (B, 1, T)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]
    


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., out_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.out_layer = out_layer() if out_layer is not None else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # Fills elements of attn with -1e9 where mask==0. 
        attn = self.dropout(F.softmax(attn, dim=-1))  # nn.Dropout(p=0.1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attention = ScaledDotProductAttention(temperature=qk_scale or head_dim ** 0.5)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, mask=None):
        B, Nq, Nk, Nv, C = q.shape[0], q.shape[1], k.shape[1], v.shape[1], q.shape[2]
        if self.with_qkv:
            q = self.proj_q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, 8, T, D/8)
            k = self.proj_k(k).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = self.proj_v(v).reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        else:
            q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)

        x, attn = self.attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).reshape(B, Nq, C)  # （B, T, D）
        if self.with_qkv:
           x = self.proj(x)  # （B, T, D)
           x = self.proj_drop(x)  # nn.Dropout(p=0)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, mask=None):
        """ x: (B, T, D)
            mask: # (B, T)
        """
        src_mask = None
        if mask is not None:
            src_mask = get_pad_mask(mask, pad_idx=0)  # (B, 1, T)
        # transformer block
        x_norm = self.norm1(x)
        x = x + self.drop_path(self.attn(q=x_norm, k=x_norm, v=x_norm, mask=src_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.enc_dec_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, memory, memory_mask=None, trg_mask=None):  # memory (B, 5, 512)
        tgt_2 = self.norm1(tgt)  # (B, 1, 512)
        tgt = tgt + self.drop_path(self.self_attn(q=tgt_2, k=tgt_2, v=tgt_2, mask=trg_mask))  # (B, 1, 512)
        # attention between a single future frame and 5 elems in memory
        # q: (B, 1, 512),  k: (B, 5, 512),  v: (B, 5, 512)
        tgt = tgt + self.drop_path(self.enc_dec_attn(q=self.norm2(tgt), k=memory, v=memory, mask=memory_mask))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return tgt



class Encoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, num_frames, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.):
        super(Encoder, self).__init__()
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # temporal positional encoding
        self.time_embed = PositionalEncoding(embed_dim, max_len=num_frames)
        self.time_drop = nn.Dropout(p=drop_rate)
        
        # construct encoder blocks with increasing drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
        ) for i in range(depth)])
        
        # layer normalization
        self.norm = norm_layer(embed_dim)
        
    
    def forward(self, x, mask=None):
        """ x: (B, T, D)
            mask: # (B, T)
        """
        x = self.time_embed(x)
        x = self.time_drop(x)
        
        for blk in self.encoder_blocks:
            x = blk(x, mask=mask)  # (B, T, D)
        
        x = self.norm(x)
        return x



class Decoder_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(Decoder_PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x = x + self.position_embedding[:, :x.shape[1], :]
        return x


class Decoder(nn.Module):
    def __init__(self, in_features, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 time_embed_type=None, num_frames=None):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim

        self.trg_embedding = nn.Linear(in_features, embed_dim)

        if time_embed_type is None or num_frames is None:
            time_embed_type = 'sin'
        self.time_embed_type = time_embed_type
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Decoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.decoder_blocks = nn.ModuleList([DecoderBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward(self, trg, memory, memory_mask=None, trg_mask=None):
        trg = self.trg_embedding(trg)
        trg = self.time_embed(trg)
        trg = self.time_drop(trg)  # (B, 1, D)

        for blk in self.decoder_blocks:
            trg = blk(trg, memory, memory_mask=memory_mask, trg_mask=trg_mask)

        trg = self.norm(trg)
        return trg