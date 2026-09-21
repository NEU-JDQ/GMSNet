

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from .config import TASK_CONFIG, DEFAULT_DATA_ROOT


class CrisisDataset(Dataset):
    def __init__(self,
                 root_dir=DEFAULT_DATA_ROOT,
                 task_name='task2',
                 phase='train',
                 transform=None,
                 text_processor=None):
        """Load image-text samples from a task-specific TSV split.

        Args:
            root_dir: Root directory containing TSV files and images.
            task_name: Task identifier in TASK_CONFIG.
            phase: Dataset split: train, dev, or test.
            transform: Optional image preprocessing callable.
            text_processor: Optional text tokenization callable."""
        self.root_dir = root_dir
        self.transform = transform
        self.text_processor = text_processor

        if task_name not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task_name}")

        self.task_info = TASK_CONFIG[task_name]
        self.label_map = self.task_info['label_map']

        tsv_name = f"task_{self.task_info['name']}_text_img_{phase}.tsv"
        self.tsv_path = os.path.join(root_dir,  tsv_name)
        self.data_list = self._read_tsv(self.tsv_path)
        print(f"[{phase.upper()}] Loaded {len(self.data_list)} samples from {tsv_name}")

    def _read_tsv(self, path):
        data = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"TSV file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return data

            header = lines[0].strip().split('\t')
            try:
                text_idx = header.index('tweet_text')
                img_idx = header.index('image')
                label_idx = header.index('label')
            except ValueError as e:
                raise ValueError(f"TSV is missing required columns (tweet_text, image, label). Available columns: {header}") from e

            for line in lines[1:]:
                line = line.strip()
                if not line: continue

                parts = line.split('\t')

                if len(parts) <= max(text_idx, img_idx, label_idx):
                    continue

                tweet_text = parts[text_idx]
                image_rel_path = parts[img_idx]
                label_str = parts[label_idx]

                # Retain only labels defined for the selected task.
                if label_str in self.label_map:
                    data.append({
                        'image_path': os.path.join(self.root_dir, image_rel_path),
                        'text': tweet_text,
                        'label': self.label_map[label_str]
                    })
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text_data = item['text']
        if self.text_processor:
            text_data = self.text_processor(item['text'])

        label = torch.tensor(item['label'], dtype=torch.long)

        return {
            'image': image,
            'text_tokens': text_data,  
            'label': label,
            'raw_text': item['text'],  
            'image_path': item['image_path']
        }