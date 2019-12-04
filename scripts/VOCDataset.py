from __future__ import print_function, division
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 base_dir,
                 split='train',
                 transforms=None,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.num_classes = 21
        self.transforms = transforms

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print(self.split)
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        for split in self.split:
            if split == "train":
                augmented = self.transforms(image=_img, mask=_target)
                return augmented['image'], augmented['mask'].permute(0, -1, 1, 2)
            elif split == 'val':
                augmented = self.transforms(image=_img, mask=_target)
                return augmented['image'], augmented['mask'].permute(0, -1, 1, 2)

    def _make_img_gt_point_pair(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        _target = np.array(Image.open(self.categories[index]))

        _target[_target == 255] = 0
        _target = np.eye(self.num_classes)[_target]

        return _img, _target

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from albumentations.core.composition import Compose
    from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip, Normalize
    from albumentations.pytorch import ToTensor

    transforms = Compose([HorizontalFlip(),
                          VerticalFlip(),
                          Normalize(),
                          ToTensor()])

    ds = VOCSegmentation(base_dir='/workspace/dataset/VOCdevkit/VOC2012/', split='train', transforms=transforms)
    it = iter(ds)
    img, gt = next(it)
    print(img.shape, gt.shape)
