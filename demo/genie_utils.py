from demo.predictors import GENIEPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from genie_detrex.tools.train_net import modify_cfg
from genie_detrex.demo.demo import get_parser


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_detrex_inference_model(custom_settings):
    args = get_parser().parse_args(["--config-file", custom_settings['config_file']])
    # args.config_file=custom_settings['config_file']
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)
    cfg, _ = modify_cfg(cfg, custom_settings)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    model = GENIEPredictor(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format
    )

    return model


if __name__ == "__main__":
    settings = dict(config_file="/content/genie_detrex/projects/dino/configs/dino-resnet/dino_r50_4scale_24ep.py",
                init_checkpoint="/content/model_0000020.pth",
                num_classes=9,
                cache_dir="/content"
                )
    model = get_detrex_inference_model(settings)
    print(model("/content/kitti_dataset/images/000001.png", return_type="all", as_numpy=True))
    print(model("/content/kitti_dataset/images/000001.png", return_type="all_probs", as_numpy=True).shape)
    print(model("/content/kitti_dataset/images/000001.png", return_type="embeddings", as_numpy=True).shape)