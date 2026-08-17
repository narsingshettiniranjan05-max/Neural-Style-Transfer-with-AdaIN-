from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.files = []

        for path, _, filenames in os.walk(root):
            for file in filenames:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files.append(os.path.join(path, file))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        try:
            image = Image.open(self.files[idx])
            image.load()
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Skipping corrupted image: {self.files[idx]}")
            
            # Try another image
            return self.__getitem__((idx + 1) % len(self.files))

def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize((size, size)))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1,1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False)+ eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std