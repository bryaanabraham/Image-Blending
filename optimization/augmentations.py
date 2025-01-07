import torch
from torch import nn
from torchvision.transforms import Compose, RandomAffine, RandomPerspective, ToTensor, Resize


class ImageAugmentations(nn.Module):
    def __init__(self, output_size, augmentations_number, p=0.7):
        super().__init__()
        self.output_size = output_size
        self.augmentations_number = augmentations_number

        self.augmentations = Compose([
            RandomAffine(degrees=15, translate=(0.1, 0.1), fill=0),
            RandomPerspective(distortion_scale=0.7, p=p),
        ])

        self.resize = Resize((self.output_size, self.output_size))

    def forward(self, input):
        """Extends the input batch with augmentations

        If the input consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]

        Args:
            input (torch.Tensor): input batch of shape [batch, C, H, W]

        Returns:
            torch.Tensor: updated batch of shape [batch * augmentations_number, C, H, W]
        """
        # Resize the input images
        resized_images = torch.stack([self.resize(img) for img in input])
        resized_images = torch.tile(resized_images, dims=(self.augmentations_number, 1, 1, 1))

        batch_size = input.shape[0]
        # We want at least one non-augmented image
        non_augmented_batch = resized_images[:batch_size]

        # Apply augmentations
        augmented_batch = torch.stack([
            self.augmentations(img.permute(1, 2, 0)).permute(2, 0, 1) for img in resized_images[batch_size:]
        ])

        # Combine non-augmented and augmented batches
        updated_batch = torch.cat([non_augmented_batch, augmented_batch], dim=0)

        return updated_batch
