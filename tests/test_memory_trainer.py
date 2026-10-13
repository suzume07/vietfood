import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from memory_trainer import DesktopDetectionTrainer


class TinyDataset(Dataset):
    rect = False

    def __len__(self):
        return 32

    def __getitem__(self, index):
        return index


class TinyTrainer(DesktopDetectionTrainer):
    def build_dataset(self, dataset_path, mode="train", batch=None):
        return TinyDataset()


class LoaderMemoryRegression(unittest.TestCase):
    def test_validation_reuses_main_process_without_extra_worker_pool(self):
        trainer = TinyTrainer.__new__(TinyTrainer)
        trainer.args = SimpleNamespace(workers=2, compile=False, cache=False)
        trainer.device = torch.device("cpu")
        train = trainer.get_dataloader("unused", batch_size=4, rank=-1, mode="train")
        val = trainer.get_dataloader("unused", batch_size=4, rank=-1, mode="val")
        try:
            self.assertEqual(train.num_workers, 2)
            self.assertEqual(train.prefetch_factor, 2)
            self.assertEqual(val.num_workers, 0)
            self.assertIsNone(val.prefetch_factor)
            self.assertEqual(trainer.args.workers, 2)
            self.assertEqual(sorted(torch.cat(list(train)).tolist()), list(range(32)))
            worker_pids = [worker.pid for worker in train.iterator._workers]
            self.assertEqual(torch.cat(list(val)).tolist(), list(range(32)))
            self.assertFalse(hasattr(val.iterator, "_workers"))
            self.assertEqual(sorted(torch.cat(list(train)).tolist()), list(range(32)))
            self.assertEqual([worker.pid for worker in train.iterator._workers], worker_pids)
        finally:
            train.close()
            val.close()


if __name__ == "__main__":
    unittest.main()
