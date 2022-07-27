import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_example(img, label):
    print('Label: ', label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def get_mean_std(loader):
    channel_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channel_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def denormalization(imgs, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return imgs * stds + means


def show_batch(dl,mean_std_all_data):
    for img, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        denorm_images = denormalization(img, *mean_std_all_data)
        ax.imshow(
            make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))
        break
