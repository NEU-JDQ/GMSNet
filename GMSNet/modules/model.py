

import torch
import torch.nn as nn


import torch
import torch.nn as nn

class ModularCrisisModel(nn.Module):
    def __init__(self,
                 visual_encoder,
                 text_encoder,
                 fusion_module,
                 classifier,
                 num_classes=None,   
                 embed_dim=256):     
        """Assemble encoders, fusion, and classification heads with optional auxiliary supervision."""
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.classifier = classifier

        if num_classes is not None:
            self.aux_vis_head = nn.Linear(embed_dim, num_classes)
            self.aux_txt_head = nn.Linear(embed_dim, num_classes)
        else:
            self.aux_vis_head = None
            self.aux_txt_head = None

    def forward(self, inputs, return_features=False):
        if isinstance(inputs, dict):
            image = inputs['image']
            text_tokens = inputs['text_tokens']
        else:
            image, text_tokens = inputs

        v_feat = self.visual_encoder(image)
        t_feat = self.text_encoder(text_tokens)

        fusion_out = self.fusion_module(v_feat, t_feat)

        if isinstance(fusion_out, tuple) and len(fusion_out) == 3:
            fused_feat, v_global, t_global = fusion_out
        else:
            fused_feat = fusion_out
            v_global, t_global = None, None

        logits = self.classifier(fused_feat)

        # Return auxiliary logits when heads are available; otherwise return unimodal features.
        if return_features:
            if self.aux_vis_head is not None and v_global is not None:
                aux_v_logits = self.aux_vis_head(v_global)
                aux_t_logits = self.aux_txt_head(t_global)
                return logits, aux_v_logits, aux_t_logits
            else:
                return logits, v_global, t_global
        else:
            return logits