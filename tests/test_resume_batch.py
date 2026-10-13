import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import train


class ResumeBatchRegression(unittest.TestCase):
    def run_resume(self, extra_args):
        old_flags = (torch.backends.cudnn.benchmark,
                     torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint = Path(temp_dir) / "weights" / "last.pt"
                checkpoint.parent.mkdir()
                checkpoint.touch()
                model = MagicMock()
                argv = ["train.py", "--resume", str(checkpoint), *extra_args]
                with patch("sys.argv", argv), patch("check_gpu.check_gpu"), \
                        patch("ultralytics.YOLO", return_value=model):
                    train.main()
                return model.train.call_args.kwargs
        finally:
            (torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32) = old_flags

    def test_resume_preserves_checkpoint_batch_when_flag_omitted(self):
        options = self.run_resume([])
        self.assertNotIn("batch", options)
        self.assertTrue(options["resume"])

    def test_resume_forwards_explicit_batch_override(self):
        options = self.run_resume(["--batch", "32"])
        self.assertEqual(options["batch"], 32)
        self.assertTrue(options["resume"])


if __name__ == "__main__":
    unittest.main()
