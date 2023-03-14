import csv
import logging
from datetime import datetime

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)

def export_results(config):
    df = pd.read_csv(config.out_file, names=config.csv_columns)

    # 현재 에이전트 목록에 없으면 제외
    xs = list(config.teams.keys())
    df = df[df["agent"].apply(lambda x: x in xs)]

    # 이번주 실시한 #(runs) 최근 게임만 선택
    df = df[df["run"] > df["run"].max() - config.runs]
    run_start = df["run"].max() - config.runs + 1
    run_end = df["run"].max()

    #
    # 평균 및 median score
    #
    df_score = df[["agent", "score"]].groupby("agent")
    mean_score = df_score.mean()
    median_score = df_score.median()

    #
    # 게임 플레이 시간
    #
    df_play_time = df[["agent", "play_time"]].groupby("agent")
    mean_play_time = df_play_time.mean()

    #mean score bar graph
    names = mean_score.index.to_list()
    plt.bar(np.arange(len(names)), mean_score["score"].to_list())
    plt.xticks(np.arange(len(names)), names)
    plt.savefig(config.fig_dir / "mean_score.png")
    plt.clf()

    #median score bar graph
    plt.bar(np.arange(len(names)), median_score["score"].to_list())
    plt.xticks(np.arange(len(names)), names)
    plt.savefig(config.fig_dir / "median_score.png")
    plt.clf()

    #mean play time bar graph
    plt.bar(np.arange(len(names)), mean_play_time["play_time"].to_list())
    plt.xticks(np.arange(len(names)), names)
    plt.savefig(config.fig_dir / "mean_play_time.png")
    plt.clf()

    #
    # TABLE
    #
    kv = {
        "mean score": mean_score,
        "median score": median_score,
        "play time": mean_play_time,
    }
    summary = pd.concat(kv.values(), axis=1, sort=True)
    summary.index.name = "agent"
    summary.columns = kv.keys()
    summary = summary.sort_values(by="median score", ascending=False)

    summary.to_csv(config.summary_dir / "summary.csv")

    #
    # README 파일 업데이트
    #
    write_readme(config, run_start, run_end)

def write_readme(config, run_start, run_end):
    def csv_to_table(filename):
        buff = f"""
.. list-table::
   :header-rows: 1

"""
        with filename.open() as f:
            reader = csv.reader(f)
            for row in reader:
                for i, item in enumerate(row):
                    if i == 0:
                        buff += f"   * - {item}\n"
                    else:
                        if item.replace(".", "", 1).isdigit():
                            # float 확인
                            buff += f"     - {float(item):.3f}\n"
                        else:
                            buff += f"     - {item}\n"
        return buff

    t_current = datetime.now()
    summary_table = csv_to_table(config.summary_dir / "summary.csv")

    # README 파일 생성
    with (config.out_dir / "README.rst").open("wt") as f:
        content = f"""
NCF2022 결과
===============
**진행 현황**

.. list-table::
   :header-rows: 1
 
   * - 시작시간
     - 현재시간
     - 경과시간
     - 게임 번호
   * - {config.start_time.isoformat()}
     - {t_current.isoformat()}
     - {t_current - config.start_time}
     - {run_start}부터 {run_end}까지

**결과 요약**
{summary_table}
- 게임번호 {run_start}부터 {run_end}까지 결과(최근 게임 결과)만 사용함
- mean score: 평균 점수
- median score: 점수의 중간값
- play time: 평균 게임 플레이 시간

**Mean Score**

.. figure:: fig/mean_score.png
   :figwidth: 200

**Median Score**

.. figure:: fig/median_score.png
   :figwidth: 200

**Mean Play Time**

.. figure:: fig/mean_play_time.png
   :figwidth: 200

"""
        f.write(content)

if __name__ == "__main__":
    export_results(config)