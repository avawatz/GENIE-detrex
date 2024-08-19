from omegaconf import OmegaConf
import os

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detrex.data.genie_dataset_mapper import GENIEDatasetDETRMapper
from detrex.data.datasets.register_genie_dataset import register_genie_dataset
from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

register_genie_dataset(name="devtest_Train",
                       data_ids={"unlabelled_set": os.listdir("/content/kitti_dataset/images")[:25],
                                 "augmented_set": os.listdir("/content/kitti_dataset/images")[25:50]},
                       project_dir="/content/kitti_dataset",
                       metadata={}
                      )

register_genie_dataset(name="devtest_Val",
                       data_ids={"evaluation_set": os.listdir("/content/kitti_dataset/images")[50:75]},
                       project_dir="/content/kitti_dataset",
                       metadata={}
                      )

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="devtest_Train"),
    mapper=L(GENIEDatasetDETRMapper)(
        project_dir="/content/kitti_dataset",
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="devtest_Val", filter_empty=False),
    mapper=L(GENIEDatasetDETRMapper)(
        project_dir="/content/kitti_dataset",
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
