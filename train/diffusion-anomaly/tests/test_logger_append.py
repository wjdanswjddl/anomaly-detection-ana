"""progress.csv must survive logger.configure() across training resumes."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from guided_diffusion import logger


class TestCSVAppendAcrossConfigure(unittest.TestCase):
    def test_progress_csv_appends_on_reconfigure(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            logger.configure(dir=str(logdir), format_strs=["csv"])
            logger.logkv("step", 100)
            logger.logkv("loss", 0.5)
            logger.dumpkvs()
            logger.Logger.CURRENT.close()

            # Second configure mimics a training resume in the same OPENAI_LOGDIR.
            logger.configure(dir=str(logdir), format_strs=["csv"])
            logger.logkv("step", 200)
            logger.logkv("loss", 0.25)
            logger.dumpkvs()
            logger.Logger.CURRENT.close()

            with (logdir / "progress.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], "100")
            self.assertEqual(rows[0]["loss"], "0.5")
            self.assertEqual(rows[1]["step"], "200")
            self.assertEqual(rows[1]["loss"], "0.25")

    def test_progress_csv_adds_new_columns_without_dropping_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            logger.configure(dir=str(logdir), format_strs=["csv"])
            logger.logkv("step", 1)
            logger.logkv("loss", 0.9)
            logger.dumpkvs()
            logger.Logger.CURRENT.close()

            logger.configure(dir=str(logdir), format_strs=["csv"])
            logger.logkv("step", 2)
            logger.logkv("loss", 0.8)
            logger.logkv("val-loss_q0", 0.7)
            logger.dumpkvs()
            logger.Logger.CURRENT.close()

            with (logdir / "progress.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], "1")
            self.assertEqual(rows[1]["val-loss_q0"], "0.7")


if __name__ == "__main__":
    unittest.main()
