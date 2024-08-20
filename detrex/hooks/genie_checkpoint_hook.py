from typing import final
from detrex.checkpoint.genie_wandb_checkpointer import GENIEWandBCheckpointer
from detectron2.engine.hooks import HookBase


class GENIEWandBPeriodicCheckpointer(GENIEWandBCheckpointer, HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        if self.trainer.iter + 1 != self.max_iter:
            self.step(self.trainer.iter)

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self.step(self.trainer.iter, final_iter=True)
