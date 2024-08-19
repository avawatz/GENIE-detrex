import os
import copy
import logging
import numpy as np
import torch

from detrex.data.detr_dataset_mapper import DetrDatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class GENIEDatasetDETRMapper(DetrDatasetMapper):
    def __init__(self, project_dir: str, **kwargs):
        self.project_dir = project_dir
        super().__init__(**kwargs)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if dataset_dict["augmented"]:
            img_path = os.path.join(self.project_dir, "augmented_set", dataset_dict["file_name"])
        else:
            img_path = os.path.join(self.project_dir, "images", dataset_dict["file_name"])
        assert os.path.isfile(img_path), f"The Image {dataset_dict['file_name']} is not found."
        image = utils.read_image(img_path, format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
