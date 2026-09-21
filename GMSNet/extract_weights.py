

import os
import argparse
import torch
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
    parser = argparse.ArgumentParser(description="Extract GMSNet Modality Gating Weights")
    parser.add_argument('--task_name', type=str, default='task2')

    parser.add_argument('--model_path', type=str,
                        default='./output_gmsnet/task2/task2/best_model.pt',
                        help='Path to the trained model checkpoint')

    parser.add_argument('--image_path', type=str, default='/home/tSdu/_New_World/xzh'
                                                          '/crisiskan/datasets/settingA/data_image/california_wildfires/11_10_2017/918110217023442944_0.jpg',
                        help='Path to the input crisis image')
    parser.add_argument('--text', type=str, default='Seen From Above: Entire California Communities Reduced to Ash',
                        help='Tweet text associated with the input image')

    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')
    parser.add_argument('--embed_dim', type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading GMSNet and preprocessing components...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    eval_transform = get_transforms(mode='eval')
    num_classes = TASK_CONFIG[args.task_name]['num_classes']

    vis_enc = ConvNextVisualEncoder()
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
    text_inputs_gpu = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items()}

    inputs = {'image': image_tensor, 'text_tokens': text_inputs_gpu}

    print("Running inference and extracting modality gate values...\n")
    with torch.no_grad():
        _ = model(inputs)

        if hasattr(model.fusion_module, 'current_weight_v') and hasattr(model.fusion_module, 'current_weight_t'):
            weight_v = float(model.fusion_module.current_weight_v[0, 0])  
            weight_t = float(model.fusion_module.current_weight_t[0, 0])  

            print("=" * 50)
            print("Modality gate values")
            print("=" * 50)
            print(f"Visual gate (beta)  : {weight_v:.4f}")
            print(f"Text gate (alpha) : {weight_t:.4f}")
            print("-" * 50)

            # Normalize gates for display; these ratios are not causal feature attributions.
            total = weight_v + weight_t
            if total > 0:
                v_percent = (weight_v / total) * 100
                t_percent = (weight_t / total) * 100
                print(f"Normalized gate values -> Visual: {v_percent:.1f}% | Text: {t_percent:.1f}%")
            print("=" * 50)

            if v_percent > 70:
                print("The normalized visual gate exceeds 70%; gate values do not measure causal contribution.")
            elif t_percent > 70:
                print("The normalized text gate exceeds 70%; gate values do not measure causal contribution.")
            else:
                print("Neither normalized gate exceeds 70%; gate values do not measure causal contribution.")

        else:
            print(
                "Gate extraction failed: AGMFusion.forward must expose 'self.current_weight_t' and 'self.current_weight_v'.")


if __name__ == "__main__":
    main()