

import torch
import torch.nn as nn
import torch.nn.functional as F






class BaseFusionModule(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.output_dim = 0

    def forward(self, visual_feats, text_feats):
        raise NotImplementedError





class AttentionPooling(nn.Module):
    """Aggregate sequence features using learned attention weights."""

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.attention(x)
        return torch.sum(x * w, dim=1)


class AGMFusion(BaseFusionModule):
    def __init__(self, visual_dim, text_dim, embed_dim=256, num_heads=4, layers=1):
        """Initialize bidirectional attention and gated multimodal fusion."""
        super().__init__(visual_dim, text_dim)

        self.vis_proj = nn.Linear(visual_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        encoder_layer_v = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.vis_self_attn = nn.TransformerEncoder(encoder_layer_v, num_layers=layers)

        encoder_layer_t = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.txt_self_attn = nn.TransformerEncoder(encoder_layer_t, num_layers=layers)

        decoder_layer_img2txt = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.img2text_cross_attn = nn.TransformerDecoder(decoder_layer_img2txt, num_layers=layers)

        decoder_layer_txt2img = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.text2img_cross_attn = nn.TransformerDecoder(decoder_layer_txt2img, num_layers=layers)

        self.vis_pooling = AttentionPooling(embed_dim)
        self.txt_pooling = AttentionPooling(embed_dim)

        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),  
            nn.Sigmoid()
        )

        self.output_dim = embed_dim

    def forward(self, visual_feats, text_feats):
        v_embed = self.vis_proj(visual_feats)  

        if text_feats.dtype != self.text_proj.weight.dtype:
            text_feats = text_feats.to(dtype=self.text_proj.weight.dtype)
        t_embed = self.text_proj(text_feats)  

        # Extract unimodal features before attention for auxiliary classification.
        v_global = v_embed.mean(dim=1)  

        t_global = t_embed[:, 0, :]  


        v_intra = self.vis_self_attn(v_embed)  

        t_intra = self.txt_self_attn(t_embed)  

        # Apply bidirectional cross-attention to the unimodal feature sequences.
        v_fused = self.img2text_cross_attn(tgt=v_intra, memory=t_intra)  

        t_fused = self.text2img_cross_attn(tgt=t_intra, memory=v_intra)  


        v_pool = self.vis_pooling(v_fused)  

        t_pool = self.txt_pooling(t_fused)  

        concat_feat = torch.cat([v_pool, t_pool], dim=1)  

        # Independent sigmoid gates are not constrained to sum to one.
        gates = self.gate_network(concat_feat)  


        alpha = gates[:, 0:1]  
        beta = gates[:, 1:2]  

        # Store detached gate values for inspection without retaining the computation graph.
        self.current_weight_t = alpha.detach().cpu().numpy()

        self.current_weight_v = beta.detach().cpu().numpy()

        final_fused_feat = alpha * t_pool + beta * v_pool  

        return final_fused_feat, v_global, t_global