from cvcore import Hook, HOOKS, build_from_cfg, get_event_storage
from ..utils import EventWriter
from .builder import WRITERS

__all__ = ['PeriodicWriterHook']


@HOOKS.register_module()
class PeriodicWriterHook(Hook):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers, interval=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            interval (int):
        """
        if not isinstance(writers, list):
            writers = [writers]

        writers_obj = []
        for w in writers:
            if isinstance(w, dict):
                w = build_from_cfg(w, WRITERS)
            else:
                assert isinstance(w, EventWriter), w
            writers_obj.append(w)

        self._writers = writers_obj
        if isinstance(interval,dict):
            self._train_interval=interval['train']
            self._val_interval=interval['val']
        else:
            self._train_interval = interval
            self._val_interval = -1

    def before_run(self, runner):
        if self._train_interval <= 0 and self._val_interval<=0:
            return
        for writer in self._writers:
            writer.init(runner)

    def after_train_iter(self, runner):
        if self._train_interval <= 0:
            return
        if (runner.iter + 1) % self._train_interval == 0 or (
                runner.iter == runner.max_iters - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_val_iter(self, runner):
        if self._val_interval<=0:
            return
        if (runner.val_iter + 1) % self._val_interval == 0:
            for writer in self._writers:
                writer.write()

    def after_iter(self, runner):
        self.after_val_iter(runner)


    def after_run(self, runner):
        if self._train_interval <= 0:
            return
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()
