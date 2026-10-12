"""Keep Windows DataLoader process count and prefetch memory bounded."""
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


class DesktopDetectionTrainer(DetectionTrainer):
    loader_prefetch = 2

    def __init__(self, *args, loader_prefetch=2, **kwargs):
        if loader_prefetch < 1:
            raise ValueError("loader_prefetch must be >= 1")
        self.loader_prefetch = loader_prefetch
        super().__init__(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        workers = self.args.workers
        try:
            # The stock detection trainer uses workers * 2 for validation.
            # Its pool coexists with persistent train workers between epochs.
            if mode == "val":
                self.args.workers = 0
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        finally:
            self.args.workers = workers

        if loader.num_workers:
            # Current InfiniteDataLoader starts workers lazily, so this takes
            # effect before its first iteration. Reset older eager iterators.
            loader.prefetch_factor = self.loader_prefetch
            if loader.iterator is not None:
                loader.reset()
        LOGGER.info(
            f"Host memory: {mode} workers={loader.num_workers}, "
            f"prefetch={loader.prefetch_factor}, cache={self.args.cache}"
        )
        return loader
