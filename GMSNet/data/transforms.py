

from PIL import Image
import torchvision.transforms as transforms


def expand2square(pil_img, background_color):
    """Pad an image to a square while preserving its aspect ratio."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def pad_to_square(image):
    """Top-level callable so Windows DataLoader workers can pickle transforms."""
    return expand2square(image, (0, 0, 0))


def get_transforms(mode='train', image_size=224):
    """Build image preprocessing with augmentation for training mode."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Pad before resizing to preserve the image aspect ratio.
    base_transforms = [
        transforms.Lambda(pad_to_square),
        transforms.Resize((image_size, image_size)),
    ]

    if mode == 'train':
        base_transforms.extend([
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomCrop((image_size, image_size)),  
            transforms.ToTensor(),
            normalize
        ])
    else:
        base_transforms.extend([
            transforms.ToTensor(),
            normalize
        ])

    return transforms.Compose(base_transforms)
