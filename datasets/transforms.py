import random
import torchvision.transforms as transforms
from PIL import Image


def get_train_transforms(image_size=256):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_val_transforms(image_size=256):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class PairedTransforms:
    def __init__(self, image_size=256, augment=False):
        self.image_size = image_size
        self.augment = augment
        self.do_flip = None

    def __call__(self, *images):
        if self.do_flip is None:
            self.do_flip = random.random() < 0.5 if self.augment else False

        tensors = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img) if hasattr(img, 'shape') else img

            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

            if self.do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            tensor = transforms.ToTensor()(img)
            tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
            tensors.append(tensor)

        self.do_flip = None
        return tuple(tensors) if len(tensors) > 1 else tensors[0]
