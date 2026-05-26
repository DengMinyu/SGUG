import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from Utilities.CUDA_Check import GPUorCPU
from torchvision.io import read_image, ImageReadMode


DEVICE = GPUorCPU().DEVICE


def load_image_safe(image_path, mode=ImageReadMode.RGB):
    """
    安全加载图像，支持多种格式（JPEG, PNG, BMP, TIF等）
    优先使用torchvision.read_image，失败时使用PIL作为备选
    """
    try:
        # 尝试使用torchvision读取（速度快，但只支持JPEG/PNG）
        img = read_image(image_path, mode=mode)
        return img
    except (RuntimeError, OSError):
        # 如果torchvision失败，使用PIL读取（支持更多格式）
        pil_img = Image.open(image_path).convert('RGB' if mode == ImageReadMode.RGB else 'L')
        # 转换为numpy数组，然后转为torch tensor
        img_array = np.array(pil_img)
        
        # PIL图像是HWC格式，需要转换为CHW格式
        if len(img_array.shape) == 3:  # RGB图像
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        elif len(img_array.shape) == 2:  # 灰度图像
            img_array = img_array[np.newaxis, :, :]  # H, W -> 1, H, W
        
        # 转换为torch tensor（uint8类型，与read_image一致）
        img = torch.from_numpy(img_array).contiguous()
        return img
model_input_image_size_height = 256
model_input_image_size_width = 256
random_crop_size = 224

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class DataLoader_Train(Dataset):
    train_valid_transforms = transforms.Compose(
        [
            transforms.Resize((model_input_image_size_height, model_input_image_size_width), antialias=False),
            transforms.RandomCrop(random_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ZeroOneNormalize(),
        ]
    )

    train_valid_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B, file_list_GT, file_list_DM):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.file_list_GT = file_list_GT
        self.file_list_DM = file_list_DM
        self.transform1 = self.train_valid_transforms
        self.transform2 = self.train_valid_transforms_Norm

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT) == len(self.file_list_DM):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        seed = torch.random.seed()

        imgA_path = self.file_list_A[idx]
        img_A = load_image_safe(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_A = self.transform1(img_A)
        imgA_transformed = self.transform2(img_A)

        imgB_path = self.file_list_B[idx]
        img_B = load_image_safe(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_B = self.transform1(img_B)
        imgB_transformed = self.transform2(img_B)

        imgGT_path = self.file_list_GT[idx]
        img_GT = load_image_safe(imgGT_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        imgGT_transformed = self.transform1(img_GT)

        imgDM_path = self.file_list_DM[idx]
        img_DM = load_image_safe(imgDM_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgDM_transformed = self.transform1(img_DM)

        return imgA_transformed, imgB_transformed, imgGT_transformed, imgDM_transformed


class Dataloader_Eval(Dataset):
    eval_transforms = transforms.Compose(
        [
            ZeroOneNormalize(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform1 = self.eval_transforms
        self.transform2 = self.eval_transforms

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        imgA_path = self.file_list_A[idx]
        img_A = load_image_safe(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        imgA_transformed = self.transform1(img_A).to(DEVICE)

        imgB_path = self.file_list_B[idx]
        img_B = load_image_safe(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        imgB_transformed = self.transform1(img_B).to(DEVICE)

        return imgA_transformed, imgB_transformed
