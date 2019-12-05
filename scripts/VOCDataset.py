from __future__ import print_function, division
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import scripts.custom_transforms as tr
from torchvision import transforms


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 base_dir,
                 split='train',
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
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=513, crop_size=513),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':

    ds = VOCSegmentation(base_dir='/workspace/dataset/VOCdevkit/VOC2012/', split='train')
    it = iter(ds)
    img, gt = next(it)
    print(img.shape, gt.shape)
