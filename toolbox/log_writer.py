import os

os.environ["GKSwstype"] = "nul"
os.environ["GKS_VIDEO_OPTS"] = "640x480@24"

import logging
import pathlib
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterable

import gr
import numpy as np
import pandas as pd
import plotille
import torch
import torch.nn as nn
import tqdm
from gr.pygr import mlab
from torch.utils.tensorboard import SummaryWriter

from toolbox.keyboard import esc_pressed

logger = logging.getLogger(__name__)


class LogWriter:
    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ):
        self._writer = SummaryWriter(
            log_dir=log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        logger.info(f"Writer 생성: {self.log_dir}")

        self.record_tag = None
        self.records = []
        self.stats = defaultdict(list)
        self._epoch = 0
        self._mean_score = -1e10
        self._highest_mean_score = -1e10
        self._new_record = False

        # 백업
        backup_file = pathlib.Path(self.log_dir) / "backup.zip"
        with zipfile.ZipFile(backup_file, "w") as zf:
            files = list(pathlib.Path().glob("**/*.py"))
            for fil in tqdm.tqdm(files, desc="Backup"):
                zf.write(fil)

    def to_table(self, kv):
        return pd.DataFrame(data=[kv]).T

    def add_table(self, tag, kv):
        self._writer.add_text(tag, self.to_table(kv).to_html())

    def add_stat(self, tag, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, Iterable):
            self.stats[tag].extend(value)
        else:
            self.stats[tag].append(value)

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def flush(self, frames):
        stats = self.stats
        scalars = {}
        # tensorboard 기록
        for k, v in stats.items():
            if len(v) > 0:
                mean_value = np.mean(v)
                self._writer.add_scalar(k, mean_value, global_step=frames)
                scalars[k] = mean_value
        logger.info(str(self.to_table(scalars)))

        # csv 저장
        csv_file = pathlib.Path(self.log_dir) / "scalars.csv"
        if not csv_file.exists():
            self.scalar_names = [n for n in stats]
            line = "frames," + ",".join(self.scalar_names) + "\n"
            csv_file.write_text(line)

        with csv_file.open("at") as f:
            line = (
                f"{frames},"
                + ",".join([str(scalars[n]) for n in self.scalar_names])
                + "\n"
            )
            f.write(line)

        scores = stats.get("score/score")
        if scores is not None and len(scores) > 0:
            bins = min(20, len(scores) // 5)
            logger.info(plotille.histogram(scores, bins=bins, height=12, lc="green",))

        logger.info(f"epoch: {self._epoch}, logdir: {self.log_dir} ")

        # 남겨둬야 할 기록 갱신
        mean_score = np.mean(scores)
        self._new_record = mean_score > self._highest_mean_score
        self._highest_mean_score = max(mean_score, self._highest_mean_score)
        self._mean_score = mean_score

        # 다음 epoch을 위해 기록 초기화
        self._epoch += 1
        self.stats.clear()
        return self._new_record

    @property
    def epoch(self):
        return self._epoch

    @property
    def mean_score(self):
        return self._mean_score

    @property
    def new_record(self):
        return self._new_record

    @property
    def log_dir(self):
        return self._writer.log_dir

    def _save_state_dict(self, tag: str, name: str, state_dict):
        model_path = pathlib.Path(self.log_dir) / tag / f"{name}.pt"
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, model_path)
        return model_path

    def _load_state_dict(self, tag: str, name: str, model: nn.Module):
        model_path = pathlib.Path(self.log_dir) / tag / f"{name}.pt"
        model.load_state_dict(torch.load(model_path))
        return model_path

    def save_model(self, name: str, model: nn.Module):
        model_path = self._save_state_dict("models", name, model.state_dict())
        logger.info(f"모델 저장: {model_path}")

    def load_model(self, name: str, model: nn.Module):
        model_path = self._load_state_dict("models", name, model)
        logger.info(f"모델 로드: {model_path}")

    def esc_pressed(self):
        return esc_pressed()

    @contextmanager
    def on_record(self, name=None):
        records = []
        yield records
        if name is not None:
            path = pathlib.Path(f"{self.log_dir}") / "recordings" / f"{name}.mp4"
            path.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"records 시작: {path}")
            gr.beginprint(str(path))
            for img in tqdm.tqdm(records):
                mlab.imshow(img)
            gr.endprint()
            logger.info(f"records 종료: {path}")
