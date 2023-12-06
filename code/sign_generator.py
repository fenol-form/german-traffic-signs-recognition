import torch
import torchvision
import torch.nn.functional as F

import torchvision.transforms.functional

import os
import tqdm

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from torchvision.io import read_image, write_png
from skimage.color import rgb2hsv, hsv2rgb


class SignGenerator(object):
    """
    Class for synthetic data generation
    :param background_path: path to folder that contains background images
    """

    def __init__(self, background_path):
        self.background_path = background_path
        self.bg_filenames = os.listdir(background_path)
        self.n_images = len(self.bg_filenames)

    def get_sample(self, icon, eps=10.):
        """
        Function that embedds an icon in random background
        :param icon: icon image's tensor
        """
        random_bg_idx = np.random.randint(low=0, high=self.n_images)
        path_to_bg = os.path.join(self.background_path, self.bg_filenames[random_bg_idx])
        bg = read_image(path_to_bg)

        # crop box size
        cropbox_size = icon.shape[1]
        assert bg.shape[1] > cropbox_size
        assert bg.shape[2] > cropbox_size
        top_ = np.random.randint(low=0, high=bg.shape[1] - cropbox_size)
        left_ = np.random.randint(low=0, high=bg.shape[2] - cropbox_size)

        # crop background
        cropped_bg = torchvision.transforms.functional.crop(bg,
                                                            top_,
                                                            left_,
                                                            cropbox_size,
                                                            cropbox_size).type(torch.float) / 255.
        icon, mask = icon[:-1], icon[-1]
        mask = torch.tile(mask.unsqueeze(0), (3, 1, 1))
        icon = torch.where(mask > eps, icon, cropped_bg)

        return icon.clip(0., 1.)


def generate_one_icon(args):
    """
    Generate synthetic for one class
    :param args: [path to file with an icon, path to output folder, path to backgrounds folder, num of examples per class]
    """
    assert len(args) == 4
    path_to_icon = args[0]
    path_to_output_folder = args[1]
    path_to_backgrounds = args[2]
    samples_per_class = args[3]

    samples = []
    for _ in range(samples_per_class):
        # read icon
        img = read_image(path_to_icon)
        if img.shape[0] == 2:
            img, mask = img[0], img[1]
            img = torch.tile(img, (3, 1, 1))
            img = torch.concatenate([img, mask.unsqueeze(axis=0)], dim=0)

        # resize
        size = np.random.randint(low=16, high=129)
        resizer = torchvision.transforms.Resize((size, size), antialias=True)
        img = resizer(img)

        # padding
        padding = int(np.random.rand() / 6.666666666666667 * size / 2)
        img = F.pad(input=img, pad=(padding, padding, padding, padding, 0, 0), mode="constant", value=0)

        # change color
        # convert img to HSV
        mask, img = img[-1], img[:-1]
        img = img.permute(1, 2, 0)
        hsv_img = rgb2hsv(img.numpy()).astype(np.float32)

        # randomly change H, S, V channels
        h_factor = 1.
        s_factor = np.random.rand() / 2. + 0.35
        v_factor = np.random.rand() / 1.5 + 0.25
        hsv_img[..., 0] = hsv_img[..., 0] * h_factor + (-0.5 + np.random.rand()) / 3.5
        hsv_img[..., 1] = hsv_img[..., 1] * s_factor + (-0.5 + np.random.rand()) / 1.5
        hsv_img[..., 2] = hsv_img[..., 2] * v_factor + (-0.5 + np.random.rand()) / 3.

        # return to rgb and torch
        img = torch.Tensor(hsv2rgb(hsv_img.clip(0., 1.))).permute(2, 0, 1)
        img = torch.concatenate([img, mask.unsqueeze(axis=0)], dim=0)

        # make kernels:
        def make_kernel(kern_size=7, makeup_iters=10):
            kernel = np.zeros((kern_size, kern_size))
            hi, wi = np.random.randint(low=0, high=kern_size, size=2)
            for _ in range(makeup_iters):
                ratio_ = np.random.rand()
                new_hi = int(hi * ratio_)
                new_wi = int(wi * ratio_)
                addition_one = np.zeros((kern_size, kern_size))
                addition_one[new_hi, new_wi] = 1.
                kernel += addition_one
            kernel /= makeup_iters
            return torch.Tensor(kernel)

        kernel = make_kernel(
            kern_size=np.random.randint(low=2, high=6) * 2 + 1,
            makeup_iters=np.random.randint(low=3, high=10)
        )
        for channel in range(4):
            img[channel] = torch.nn.functional.conv2d(
                img[channel].unsqueeze(dim=0).unsqueeze(dim=0),
                kernel.unsqueeze(dim=0).unsqueeze(dim=0),
                padding="same"
            ).squeeze(dim=(0, 1))

        # random rotate
        img = torchvision.transforms.RandomRotation(degrees=18)(img)

        # get Gausssssianed
        img = torchvision.transforms.GaussianBlur(
            kernel_size=np.random.randint(low=1, high=5) * 2 + 1,
            sigma=(0.1, 2.8)
        )(img)

        # set background
        sign_gen = SignGenerator(path_to_backgrounds)
        """
            we have to set background under the image by the values of mask
            but we have to modify mask so that it has the same position as an image after all transformations
            thus mask get some non zero and non 255. values and we should specify some EPS that will help us understand
            which elements are mask and whicn aren't 
        """
        img = sign_gen.get_sample(img, eps=img[3][img[3] > 0].min() + 100)  ### 100 is EPS

        # append
        samples.append((img * 255).type(torch.uint8))

    # save data
    head, icon_name = os.path.split(path_to_icon)
    img_class = ".".join(icon_name.split(".")[:-1])
    path_to_output_folder = os.path.join(path_to_output_folder, img_class)
    os.makedirs(name=path_to_output_folder, exist_ok=True)
    for i in range(samples_per_class):
        # generate filename
        fn = icon_name.split(sep=".")
        fn.append(str(i))
        fn[-1], fn[-2] = fn[-2], fn[-1]
        fn = ".".join(fn)
        # save
        write_png(samples[i], os.path.join(path_to_output_folder, fn))


def generate_all_data(output_folder, icons_path, background_path, samples_per_class=1000):
    """
    :param output_folder: path to output folder
    :param icons_path: path to folder that contains signs
    :param background_path: path to backgrounds folder
    :param samples_per_class: num of examples per each class to be generated
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))
