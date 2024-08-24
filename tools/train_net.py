#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import math
import json
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.checkpoint import DetectionCheckpointer
from detrex.hooks.genie_checkpoint_hook import GENIEWandBPeriodicCheckpointer
# from detrex.checkpoint import DetectionCheckpointer

from detrex.utils import WandbWriter
from detrex.modeling import ema

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])



def _format_output(self, table, columnns):
    contents = dict()

    contents["columns"] = columnns
    contents["index"] = list()
    contents["data"] = list()

    for index, row in enumerate(table):
        contents["index"].append(index)
        contents["data"].append(row)

    return contents


def write_results(results):
    assert results.get('bbox', None) is not None
    results = results['bbox']
    


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
            print("RET", ret)
        return ret
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    
    # instantiate optimizer
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # build training loader
    train_loader = instantiate(cfg.dataloader.train)
    
    # create ddp model
    model = create_ddp_model(model, **cfg.train.ddp)

    # build model ema
    ema.may_build_model_ema(cfg, model)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )
    
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier))
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            GENIEWandBPeriodicCheckpointer(checkpointer, **cfg.train.checkpointer),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    print(trainer._hooks)

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


from detrex.data.datasets.register_genie_dataset import register_genie_dataset
from detrex.data.genie_dataset_mapper import GENIEDatasetDETRMapper
from detrex.evaluation.genie_evaluator import GENIECOCOEvaluator


def modify_cfg(cfg, custom_args: dict):
    if custom_args.get("data_ids", None) is not None:
        register_genie_dataset(name="genie_test",
                               data_ids={"evaluation_set": custom_args["data_ids"].pop("evaluation_set")},
                               project_dir=custom_args["project_dir"],
                               num_classes=custom_args["num_classes"],
                               metadata={})

        register_genie_dataset(name="genie_train",
                               data_ids=custom_args["data_ids"],
                               project_dir=custom_args["project_dir"],
                               num_classes=custom_args["num_classes"],
                               metadata={})

        cfg.dataloader.train.dataset.names = "genie_train"
        cfg.dataloader.train.mapper['project_dir'] = custom_args["project_dir"]
        cfg.dataloader.train.mapper._target_ = GENIEDatasetDETRMapper
        cfg.dataloader.train.total_batch_size = custom_args["total_batch_size"]
        cfg.dataloader.train.num_workers = custom_args["num_workers"]

        cfg.dataloader.test.dataset.names = "genie_test"
        cfg.dataloader.test.mapper['project_dir'] = custom_args["project_dir"]
        cfg.dataloader.test.mapper._target_ = GENIEDatasetDETRMapper
        cfg.dataloader.test.num_workers = custom_args["num_workers"]

        cfg.dataloader.evaluator["output_dir"] = custom_args["cache_dir"]
        cfg.dataloader.evaluator._target_ = GENIECOCOEvaluator

    if custom_args.get("wandb", None) is not None:
        wandb_cache_dir = os.path.join(custom_args["cache_dir"], "wandb_cache")
        os.makedirs(wandb_cache_dir, exist_ok=True)
        cfg.train.wandb.enabled = True
        cfg.train.wandb.params.project = custom_args["wandb"]["project"]
        cfg.train.wandb.params.name = None
        cfg.train.wandb.dir = wandb_cache_dir
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
        os.environ["WANDB_API_KEY"] = custom_args["wandb"]["api_key"]

    if custom_args.get("init_checkpoint", None) is not None:
        cfg.train.init_checkpoint = custom_args["init_checkpoint"]
    cfg.train.output_dir = custom_args["cache_dir"]
    cfg.train.checkpointer.period = cfg.train.eval_period

    if custom_args.get("num_classes", None) is not None:
        if cfg.model.get("num_classes", None) is not None:
            cfg.model.num_classes = custom_args["num_classes"]
        if cfg.model.criterion.get("num_classes", None) is not None:
            cfg.model.criterion.num_classes = custom_args["num_classes"]

    return cfg, custom_args.get("eval_params", None)


def format_results(result, metadata):
    result = result['bbox']
    out = {"tables": list(),
           "plots": None}
    n_classes = metadata.pop('num_classes')
    table = {"columns": ["Name", "Value"],
             "index": list(),
             "data": list()}
    table['index'].append(0)
    table['data'].append(["Model ID", metadata.pop('model_id')])
    table['index'].append(1)
    table['data'].append(["Number of Classes", n_classes])
    out['tables'].append({"name": "Details of Checkpoint",
                          "data": table})

    table = {"columns": ["Metric", "Value"],
             "index": list(),
             "data": list()}
    table['index'].append(0)
    table['data'].append(['mAP@IoU.5', result.pop('mAP@IoU.5') if not math.isnan(result['mAP@IoU.5']) else 0])
    table['index'].append(1)
    table['data'].append(['mAP@IoU.75', result.pop('mAP@IoU.75') if not math.isnan(result['mAP@IoU.75']) else 0])
    table['index'].append(2)
    table['data'].append(['mAP', result.pop('mAP') if not math.isnan(result['mAP']) else 0])
    table['index'].append(3)
    table['data'].append(['mAR', result.pop('mAR') if not math.isnan(result['mAR']) else 0])
    table['index'].append(4)
    table['data'].append(['mAR@Det1', result.pop('mAR@Det1') if not math.isnan(result['mAR@Det1']) else 0])
    table['index'].append(5)
    table['data'].append(['mAR@Det10', result.pop('mAR@Det10') if not math.isnan(result['mAR@Det10']) else 0])
    out['tables'].append({"name": "Common Evaluation Metrics",
                          "data": table})

    table = {"columns": ["Name", "mAP", "mAR"],
             "index": list(),
             "data": list()}
    for idx, name in enumerate(['Small', 'Medium', 'Large']):
        table['index'].append(idx)
        table['data'].append([name,
                              result.pop(f'mAP-{name}') if not math.isnan(result[f'mAP-{name}']) else 0,
                              result.pop(f'mAR-{name}') if not math.isnan(result[f'mAR-{name}']) else 0])
    out['tables'].append({"name": "Evaluation Metrics Based on Size",
                          "data": table})

    table = {"columns": ["Name", "mAP", "mAR"],
             "index": list(),
             "data": list()}
    idx = 0
    for key, val in result.items():
        if key.startswith('mAP'):
            cls_name = key.split('-')[-1]
            table['index'].append(idx)
            if math.isnan(val):
                val = 0
            table['data'].append([cls_name,
                                  val,
                                  result[f"mAR-{cls_name}"] if not math.isnan(result[f"mAR-{cls_name}"]) else 0])
            idx += 1
        else:
            continue
        if idx == n_classes:
            break
    out['tables'].append({"name": "Class-Wise Evaluation Metrics",
                          "data": table})
    return out


def main(args, custom_args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg, eval_params = modify_cfg(cfg, custom_args=custom_args)
    default_setup(cfg, args)

    # Enable fast debugging by running several iterations to check for any bugs.
    if True:
        cfg.train.max_iter = 20
        cfg.train.eval_period = 5
        cfg.train.log_period = 1
    cfg.train.checkpointer.period = cfg.train.eval_period

    if args.eval_only:
        assert eval_params is not None, "eval_params it is not specified"
        eval_params.update({"num_classes": custom_args["num_classes"]})
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        result = do_test(cfg, model, eval_only=True)
        result = format_results(result, eval_params)
        with open(eval_params['save_path'], 'w') as j_file:
            json.dump(result, j_file)
    else:
        do_train(args, cfg)


def custom_main(config, only_eval=False, num_gpus=1):
    args = default_argument_parser().parse_args()
    args.config_file = config["config_file"]
    args.num_machines = 1
    args.num_gpus = num_gpus
    args.eval_only = only_eval
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, config, ),
    )
