'''
Class tải dữ liệu:

Tham số: 
batch_size (int)
data_dir: đường dẫn đến directory data (Path)
auto_transforms: weights.transform() của pretrained model

Trả về:
prepare_data(): tải dữ liệu được ghi trong data_dir
setup(): chuẩn bị dữ liệu dựa trên hàm gọi 
data_loader(): trả về data_loader với data là train/test/val 
'''
import os
from pathlib import Path
from torchvision.transforms import v2 as TV2
from torch.utils.data import random_split, DataLoader, Subset
import pytorch_lightning as PL
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B2_Weights
class DataModule(PL.LightningDataModule) :
    def __init__(self, batch_size=32, data_dir = Path(r"D:\Python Project\git\DEEP LEARNING\Transfer_Learning\pizza_steak_sushi"), num_workers = os.cpu_count(), persistent_worker = False):
        super().__init__()
        auto_transform = EfficientNet_B2_Weights.DEFAULT.transforms()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.class_names = []
        self.persistent_worker = persistent_worker
        self.train_transform = TV2.Compose([
            TV2.Resize(auto_transform.resize_size, interpolation=auto_transform.interpolation),
            TV2.CenterCrop(auto_transform.crop_size),
            TV2.TrivialAugmentWide(num_magnitude_bins=31),
            TV2.ToTensor(),
            TV2.Normalize(mean=auto_transform.mean, std=auto_transform.std)
        ])
        self.test_transform = TV2.Compose([
            TV2.Resize(auto_transform.resize_size, interpolation=auto_transform.interpolation),
            TV2.CenterCrop(auto_transform.crop_size),
            TV2.ToTensor(),
            TV2.Normalize(mean=auto_transform.mean, std=auto_transform.std)
        ])
    def prepare_data(self):
        pass
    def setup(self, stage = None):
        if (stage == 'fit' or stage is None):
            full_train_ds = ImageFolder(root=self.data_dir / 'train')
            self.class_names = full_train_ds.classes
            train_len = int(0.8 * len(full_train_ds))
            val_len = len(full_train_ds) - train_len
            train_indices, val_indices = random_split(range(len(full_train_ds)), [train_len, val_len])
            train_ds_with_transform = ImageFolder(root=self.data_dir / 'train', transform=self.train_transform)
            val_ds_with_transform = ImageFolder(root=self.data_dir / 'train', transform=self.test_transform)

            self.train_ds = Subset(train_ds_with_transform, indices=train_indices.indices)
            self.val_ds = Subset(val_ds_with_transform, indices=val_indices.indices)

        if (stage == 'test' or stage is None):
            self.test_ds = ImageFolder(root = self.data_dir / 'test', transform=self.test_transform)
            self.class_names = self.test_ds.classes
    def get_class_names(self):
        return self.class_names
    def train_dataloader(self):
        return DataLoader(dataset = self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_worker)
    def test_dataloader(self):
        return DataLoader(dataset = self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_worker)
    def val_dataloader(self):
        return DataLoader(dataset = self.val_ds, batch_size=self.batch_size,num_workers=self.num_workers, persistent_workers=self.persistent_worker)
# Giả sử bạn đã import các thư viện cần thiết:
# from torchvision.datasets import ImageFolder
# from torch.utils.data import Subset, random_split
# from pathlib import Path
# import torchvision.transforms.v2 as TV2

def setup(self, stage=None):
    if stage == 'fit' or stage is None:
        full_train_ds = ImageFolder(root=self.data_dir / 'train')
        train_len = int(0.8 * len(full_train_ds))
        val_len = len(full_train_ds) - train_len
        train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])

        # Gán transform riêng cho từng split
        train_ds.dataset.transform = self.train_transform
        val_ds.dataset.transform = self.test_transform

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.class_names = full_train_ds.classes

    # Phần setup cho stage 'test' giữ nguyên hoặc tương tự
    if stage == 'test' or stage is None:
        self.test_ds = ImageFolder(root=self.data_dir / 'test', transform=self.test_transform)