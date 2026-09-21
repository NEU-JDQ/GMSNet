

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from .config import TASK_CONFIG, DEFAULT_DATA_ROOT
from .protocol import read_rows, resolve_image


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
        return [{'image_path': str(resolve_image(self.root_dir, row['image'])),
                 'text': row['tweet_text'], 'label': self.label_map[row['label']]}
                for row in read_rows(path, self.label_map)]

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