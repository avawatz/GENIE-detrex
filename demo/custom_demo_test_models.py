# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import warnings
import cv2
import tqdm

sys.path.insert(0, "./")  # noqa
from demo.predictors import GENIEPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from tools.train_net import modify_cfg


# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="detrex demo for visualizing customized inputs")
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def custom_inference_main(custom_settings):
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    args.metadata_dataset = "genie_test"
    args.output = custom_settings["save_dir"]
    args.config_file = custom_settings["config"]
    cfg = setup(args)
    cfg = modify_cfg(cfg, custom_settings)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    demo = GENIEPredictor(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )

    if custom_settings["files"]:
        # if len(args.input) == 1:
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(custom_settings["files"]):
            # use PIL, to be consistent with evaluation
            path = os.path.join(custom_settings['project_dir'], 'images', path)
            img = read_image(path, format="BGR")
            start_time = time.time()
            print(type(demo(img)))
            print(demo(img, return_type="all_probs").shape)
            print(demo(img, return_type="embeddings").shape)
            # predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )
            # print(predictions)
            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(path))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
            #     visualized_output.save(out_filename)
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

import os

def get_python_files_recursively(folder_path):
    python_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.py'):
                python_files.append(os.path.join(root, file_name))
    return python_files


from detectron2.data.catalog import _DatasetCatalog, _MetadataCatalog, DatasetCatalog, MetadataCatalog

if __name__ == "__main__":
    save_dir = "./test"
    os.makedirs(save_dir, exist_ok=True)
    for name in os.listdir("/content/GENIE-detrex/projects"):
        if name in ["sqr_detr", "co_mot", "maskdino"]:
            continue
        configs = get_python_files_recursively(f"/content/GENIE-detrex/projects/{name}/configs")
        if not len(configs):
            continue
        for config in tqdm.tqdm(configs):
            if os.path.basename(config) in ['anchor_detr_r50.py', 'h_deformable_detr_r50.py', 'timm_example.py', 'torchvision_example.py',
            'focalnet.py', 'train_net.py', 'dino_eva_02_vitdet_l_4attn_1024_lrd0p8_4scale_12ep.py', 'dino_eva_02_vitdet_l_4attn_1280_lrd0p8_4scale_12ep.py',
            'dino_eva_02_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.py'] or os.path.dirname(config).endswith("models") or os.path.dirname(config).endswith("common") or \
            os.path.dirname(config).endswith("scheduler") or os.path.dirname(config).endswith("data"):
                continue
            DatasetCatalog = _DatasetCatalog()
            MetadataCatalog = _MetadataCatalog()
            print(f"{name} - {os.path.basename(config)}")
            cfg = {"cache_dir": "/content/test_cache",
                "files": os.listdir("/content/kitti_dataset/images")[:1],
                "project_dir": "/content/kitti_dataset",
                "total_batch_size": 5,
                "num_workers": 2,
                "config": config,
                "num_classes": 80,
                "init_checkpoint": "",
                "save_dir": save_dir}
            custom_inference_main(custom_settings=cfg)
