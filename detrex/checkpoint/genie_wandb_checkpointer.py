import wandb
from fvcore.common.checkpoint import PeriodicCheckpointer
from typing import Any
import os
from detectron2.utils.events import get_event_storage, has_event_storage

class GENIEWandBCheckpointer(PeriodicCheckpointer):
    def __init__(self, checkpointer, wandb_sync=True, **kwargs):
        self.wandb_sync = wandb_sync
        self.best_metric = -1
        super().__init__(checkpointer, **kwargs)

    def _check_ckpt(self, iteration):
        assert has_event_storage(), "Event Storage not Found"
        storage = get_event_storage().latest().copy()
        if storage.get('bbox/mAP@IoU.5', None) is not None:
            key = 'bbox/mAP@IoU.5'
        elif storage.get('segment/mAP@IoU.5', None) is not None: # The Key needs to be corrected
            key = 'segment/mAP@IoU.5'
        else:
            raise ValueError("The metric is not found at the Event storage. Always Run EvalHook before the running of this Hook.")
        assert storage.get(key)[-1] == iteration, f"The Latest evaluation iteration {storage.get(key)[-1]} doesn't match the current iteration {iteration}."
        if self.best_metric < storage[key][0]:
            self.best_metric = storage[key][0]
            for key in list(storage.keys()):
                if key.startswith('bbox') or key.startswith('segment'): # The Key needs to be corrected
                    storage[key] = storage[key][0]
                else:
                    del storage[key]
            return True, storage
        else:
            return False, storage

    def _log_ckpt_at_wandb(self, ckpt_path, iteration, metadata):
        model_checkpoint_artifact = wandb.Artifact(f"run_{wandb.run.id}_model", 
                                    "model",
                                    metadata=metadata)
        model_checkpoint_artifact.add_file(ckpt_path)
        wandb.log_artifact(
            model_checkpoint_artifact, aliases=[f"itr-{iteration + 1} mAP-{self.best_metric}", "latest"]
        )

    def step(self, iteration: int, final_iter: bool = False, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        print(iteration, self.max_iter)

        if (iteration + 1) % self.period == 0 or final_iter:
            self.checkpointer.save(
                "{}_{:07d}".format(self.file_prefix, iteration), **additional_state
            )
            ckpt_status, metadata = self._check_ckpt(iteration=iteration)
            if ckpt_status:
                print("LOGGED THE CHECKPOINT")
            else:
                print("NOT LOGGED THE CHECKPOINT")
            if self.wandb_sync and wandb.run is not None and ckpt_status:
                ckpt_path = os.path.join(self.checkpointer.save_dir, "{}_{:07d}.pth".format(self.file_prefix, iteration))
                self._log_ckpt_at_wandb(ckpt_path=ckpt_path, iteration=iteration, metadata=metadata)

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        # if self.max_iter is not None:
        #     if iteration >= self.max_iter - 1:
        #         self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)
        #         ckpt_status, metadata = self._check_ckpt(iteration=iteration)
        #         if ckpt_status:
        #             print("LOGGED THE CHECKPOINT")
        #         else:
        #             print("NOT LOGGED THE CHECKPOINT")
        #         if self.wandb_sync and wandb.run is not None and ckpt_status:
        #             ckpt_path = os.path.join(self.save_dir, "{}_{:07d}.pth".format(self.file_prefix, iteration))
        #             self._log_ckpt_at_wandb(ckpt_path=ckpt_path, iteration=iteration)
