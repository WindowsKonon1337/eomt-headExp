import csv
from pathlib import Path
from typing import Dict

import torch
from lightning.pytorch.callbacks import Callback


class CSVMetricsCallback(Callback):
    def __init__(self, output_path: str = "logs/metrics.csv"):
        super().__init__()
        self.output_path = Path(output_path)
        self._header_written = False

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _write_metrics(self, trainer, phase: str, metrics: Dict[str, object]):
        if not trainer.is_global_zero:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, value in metrics.items():
            value_float = self._to_float(value)
            if value_float is None:
                continue
            rows.append(
                {
                    "global_step": trainer.global_step,
                    "epoch": trainer.current_epoch,
                    "phase": phase,
                    "metric": key,
                    "value": value_float,
                }
            )

        if not rows:
            return

        with self.output_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["global_step", "epoch", "phase", "metric", "value"],
            )
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = {
            k: v
            for k, v in trainer.logged_metrics.items()
            if k.startswith("losses/train_") or k.startswith("attn_mask_prob_")
        }
        self._write_metrics(trainer, "train", metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            k: v
            for k, v in trainer.callback_metrics.items()
            if k.startswith("metrics/val_") or k.startswith("losses/val_")
        }
        self._write_metrics(trainer, "val", metrics)
