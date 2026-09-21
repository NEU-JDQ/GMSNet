import torch
import torch.nn.functional as F
from PIL import Image
import cv2  
import numpy as np  


from data import TextProcessor, get_transforms
from modules import (
    ConvNextVisualEncoder,
    DebertaTextEncoder,
    AGMFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


class CrisisClassificationModel:
    def __init__(self, weight_path, num_classes, text_model_path='../local_models/deberta-v3-base'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        print(f"Initializing model components for {num_classes} classes...")

        vis_enc = ConvNextVisualEncoder()
        txt_enc = DebertaTextEncoder(model_path=text_model_path)

        fusion = AGMFusion(
            visual_dim=vis_enc.output_dim, text_dim=txt_enc.output_dim,
            embed_dim=256, num_heads=4, layers=1
        )
        cls_head = CrisisKANClassifier(
            input_dim=fusion.output_dim, num_classes=num_classes, dropout_rate=0.4
        )

        self.model = ModularCrisisModel(
            vis_enc, txt_enc, fusion, cls_head,
            num_classes=num_classes, embed_dim=256
        ).to(self.device)

        print(f"Loading checkpoint from: {weight_path}")
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.text_processor = TextProcessor(model_name=text_model_path)
        self.test_transforms = get_transforms(mode='eval')

        self.attn_weights = None

        def cross_attn_hook(module, input, output):
            try:
                query = input[0].detach()
                key = input[1].detach()

                # Compute an unprojected similarity proxy, not the exact multi-head attention weights.
                scores = torch.bmm(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
                weights = F.softmax(scores, dim=-1)

                self.attn_weights = weights.cpu().numpy()[0]
            except Exception as e:
                print(f"Attention hook computation failed: {e}")

        try:
            self.model.fusion_module.text2img_cross_attn.layers[0].multihead_attn.register_forward_hook(cross_attn_hook)
        except Exception as e:
            print(f"Attention hook registration failed: {e}")

        print("Model and heatmap components initialized.")

    def predict(self, image_file, text):
        """Return class probabilities and an optional image heatmap."""
        orig_img_pil = Image.open(image_file).convert('RGB')
        image_tensor = self.test_transforms(orig_img_pil).unsqueeze(0).to(self.device)

        text_inputs_raw = self.text_processor(text)
        text_inputs = {}
        for k, v in text_inputs_raw.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            text_inputs[k] = v.to(self.device)

        inputs = {'image': image_tensor, 'text_tokens': text_inputs}

        with torch.no_grad():
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        heatmap_img = None
        try:
            if self.attn_weights is not None:
                orig_img_cv = cv2.cvtColor(np.array(orig_img_pil), cv2.COLOR_RGB2BGR)
                real_h, real_w = orig_img_cv.shape[:2]

                overall_attn_raw = np.mean(self.attn_weights, axis=0)  
                overall_attn = overall_attn_raw.reshape(7, 7)

                overall_attn = overall_attn - np.min(overall_attn)
                if np.max(overall_attn) > 0:
                    overall_attn = overall_attn / np.max(overall_attn)

                overall_heatmap = cv2.resize(overall_attn, (real_w, real_h))
                overall_heatmap = np.uint8(255 * overall_heatmap)
                heatmap_color = cv2.applyColorMap(overall_heatmap, cv2.COLORMAP_JET)

                superimposed_img = heatmap_color * 0.4 + orig_img_cv * 0.6
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

                heatmap_img = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Heatmap generation failed: {e}")

        return probs, heatmap_img