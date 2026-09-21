
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig
from collections import OrderedDict
from transformers import AutoModel, AutoConfig, AutoTokenizer
from timm.models import load_checkpoint







class BaseVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, images):
        """Return a sequence of features with shape (batch, sequence, dimension)."""
        raise NotImplementedError


class BaseTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, text_inputs):
        """Return a sequence of features with shape (batch, sequence, dimension)."""
        raise NotImplementedError






class DenseNetVisualEncoder(BaseVisualEncoder):
    def __init__(self, weights_path='../local_models/densenet201-c1103571.pth', pretrained=False):
        super().__init__()

        self.backbone = models.densenet201(pretrained=pretrained)
        self.output_dim = 1920

        if weights_path:
            self._load_local_weights(weights_path)

        self.dropout = nn.Dropout(0.1)

    def _load_local_weights(self, path):
        print(f"[VisualEncoder] Loading weights from: {path}")
        try:
            loaded_state = torch.load(path, map_location='cpu')
            if isinstance(loaded_state, dict):
                if 'state_dict' in loaded_state:
                    loaded_state = loaded_state['state_dict']
                elif 'model' in loaded_state:
                    loaded_state = loaded_state['model']

            new_state_dict = OrderedDict()
            model_keys = list(self.backbone.state_dict().keys())

            for k, v in loaded_state.items():
                name = k.replace('module.', '')
                name = name.replace('norm.1', 'norm1').replace('norm.2', 'norm2')
                name = name.replace('conv.1', 'conv1').replace('conv.2', 'conv2')

                if name in model_keys:
                    new_state_dict[name] = v
                elif ('features.' + name) in model_keys:
                    new_state_dict['features.' + name] = v
                else:
                    new_state_dict[name] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"[VisualEncoder] Missing keys: {len(msg.missing_keys)}")

        except Exception as e:
            print(f"[VisualEncoder] Error: {e}")

    def forward(self, images):
        features = self.backbone.features(images)
        features = F.relu(features, inplace=True)

        B, C, H, W = features.shape
        features = features.view(B, C, H * W)

        features = features.permute(0, 2, 1)

        return self.dropout(features)






class ElectraTextEncoder(BaseTextEncoder):
    def __init__(self, model_path='../local_models/google/electra-base-discriminator'):
        super().__init__()
        self.output_dim = 768

        print(f"[TextEncoder] Loading Electra from {model_path} ...")
        try:
            config = ElectraConfig()
            self.backbone = ElectraModel(config).from_pretrained(model_path)
        except Exception:
            self.backbone = ElectraModel.from_pretrained('google/electra-base-discriminator')

        self.dropout = nn.Dropout(0.1)

    def forward(self, text_inputs):
        outputs = self.backbone(**text_inputs)

        last_hidden_state = outputs.last_hidden_state

        return self.dropout(last_hidden_state)


class ResNetVisualEncoder(BaseVisualEncoder):
    def __init__(self, weights_path='', pretrained=None):
        super().__init__()
        self.backbone = models.resnet50(weights=None)

        if weights_path:
            print(f"[VisualEncoder] Loading ResNet weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
        elif pretrained:
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        self.output_dim = 2048
        self.dropout = nn.Dropout(0.1)

    def forward(self, images):
        features = self.feature_extractor(images)

        B, C, H, W = features.shape
        features = features.view(B, C, H * W)

        features = features.permute(0, 2, 1)

        return self.dropout(features)



class BERTweetTextEncoder(BaseTextEncoder):
    def __init__(self, model_path=''):
        super().__init__()
        self.output_dim = 768

        print(f"[TextEncoder] Loading BERTweet from {model_path} ...")
        self.backbone = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_inputs):
        outputs = self.backbone(**text_inputs)
        last_hidden_state = outputs.last_hidden_state
        return self.dropout(last_hidden_state)






class ConvNextVisualEncoder(nn.Module):
    def __init__(self, weights_path='../local_models/convnextv2_base/model.safetensors'):
        super().__init__()

        self.backbone = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k',
            pretrained=False,
            num_classes=0
        )

        if weights_path and os.path.exists(weights_path):
            print(f"Loading local visual weights from {weights_path}")
            load_checkpoint(self.backbone, weights_path, strict=False)
        else:
            print("Warning: Local visual weights were not found; using random initialization. Verify the weights path.")

        self.output_dim = self.backbone.num_features  

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = features.flatten(2).transpose(1, 2)
        return features




class DebertaTextEncoder(nn.Module):
    def __init__(self, model_path='../local_models/deberta-v3-base'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path)
        self.output_dim = self.backbone.config.hidden_size 

    def forward(self, text_tokens):
        input_ids = text_tokens['input_ids']
        attention_mask = text_tokens.get('attention_mask', None)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        sequence_output = outputs.last_hidden_state
        return sequence_output