import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer


from data import TextProcessor, get_transforms, TASK_CONFIG
from modules import (
    ConvNextVisualEncoder,
    DebertaTextEncoder,
    AGMFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GMSNet Cross-Attention Heatmaps")
    parser.add_argument('--task_name', type=str, default='task2')

    parser.add_argument('--model_path', type=str,
                        default=r'D:\xzh\crisiskan\new_crisiskan\output_gmsnet\task2\task2\best_model.pt',
                        help='Path to the trained model checkpoint')

    parser.add_argument('--image_path', type=str,
                        default=r'D:\xzh\crisiskan\datasets\settingA\data_image/mexico_earthquake/21_9_2017/910707671296364544_0.jpg',
                        help='Path to the input crisis image')

    parser.add_argument('--text', type=str,
                        default='Mexico earthquake â€“ Hope amid tragedy as survivors pulled from collapsed school',
                        help='Tweet text associated with the input image')

    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='./heatmaps1')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GMSNet with auxiliary classification heads...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    eval_transform = get_transforms(mode='eval')
    num_classes = TASK_CONFIG[args.task_name]['num_classes']

    vis_enc = ConvNextVisualEncoder(weights_path=None)
    txt_enc = DebertaTextEncoder(model_path=args.text_model_path)
    fusion = AGMFusion(visual_dim=vis_enc.output_dim, text_dim=txt_enc.output_dim, embed_dim=args.embed_dim)
    cls_head = CrisisKANClassifier(input_dim=fusion.output_dim, num_classes=num_classes)

    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head, num_classes=num_classes, embed_dim=args.embed_dim)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Preprocessing the input image and text...")
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = eval_transform(image).unsqueeze(0).to(device)  

    text_inputs = text_proc(args.text)
    text_inputs = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items()}

    tokens = tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])

    print("Extracting text-to-image attention weights...")
    with torch.no_grad():
        v_feat = model.visual_encoder(image_tensor)
        v_feat = v_feat.float()
        t_feat = model.text_encoder(text_inputs)
        t_feat = t_feat.float()


        v_embed = model.fusion_module.vis_proj(v_feat)
        t_embed = model.fusion_module.text_proj(t_feat)
        v_intra = model.fusion_module.vis_self_attn(v_embed)
        t_intra = model.fusion_module.txt_self_attn(t_embed)

        # Probe text-to-image attention using the first cross-attention layer.
        attn_output, attn_weights = model.fusion_module.text2img_cross_attn.layers[0].multihead_attn(
            query=t_intra,
            key=v_intra,
            value=v_intra,
            need_weights=True
        )
        attn_weights = attn_weights.squeeze(0).cpu().numpy()  

    print(f"Generating attention overlays in {args.output_dir}...")
    orig_img = cv2.imread(args.image_path)
    real_h, real_w = orig_img.shape[:2]

    valid_attn_list = []

    for idx, token in enumerate(tokens):
        clean_token = token.replace('Ġ', '').replace(' ', '').replace('##', '')
        if clean_token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', '', '.', ',', ';', ':', 'http', 'https']:
            continue

        token_attn_raw = attn_weights[idx]
        valid_attn_list.append(token_attn_raw)

        # Map 49 visual tokens to the 7 x 7 grid produced for the default input size.
        token_attn = token_attn_raw.reshape(7, 7)

        token_attn = token_attn - np.min(token_attn)
        if np.max(token_attn) > 0:
            token_attn = token_attn / np.max(token_attn)

        heatmap = cv2.resize(token_attn, (real_w, real_h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap_color * 0.4 + orig_img * 0.6

        save_path = os.path.join(args.output_dir, f"attn_word_{idx}_{clean_token}.jpg")
        cv2.imwrite(save_path, superimposed_img)
        print(f"   Saved attention overlay for token '{clean_token}' to {save_path}")

    if valid_attn_list:
        overall_attn_raw = np.mean(valid_attn_list, axis=0)
        overall_attn = overall_attn_raw.reshape(7, 7)

        overall_attn = overall_attn - np.min(overall_attn)
        if np.max(overall_attn) > 0:
            overall_attn = overall_attn / np.max(overall_attn)

        overall_heatmap = cv2.resize(overall_attn, (real_w, real_h))
        overall_heatmap = np.uint8(255 * overall_heatmap)
        overall_heatmap_color = cv2.applyColorMap(overall_heatmap, cv2.COLORMAP_JET)

        overall_superimposed = overall_heatmap_color * 0.4 + orig_img * 0.6

        overall_save_path = os.path.join(args.output_dir, "attn_overall_focus.jpg")
        cv2.imwrite(overall_save_path, overall_superimposed)
        print(f"\n   Saved mean token-attention overlay to {overall_save_path}")

    print("\nAttention visualization completed.")

if __name__ == "__main__":
    main()