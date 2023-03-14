import argparse
import copy
import datetime
import logging
import pathlib
import platform
import sys
import time
from collections import namedtuple
from functools import lru_cache, singledispatchmethod
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import toml

from toolbox.logging import configure_logger

logger = logging.getLogger(__name__)


def add_datetime(s: str):
    now = datetime.datetime.now()
    return f'runs/{s}-{now.strftime("%y-%m-%d-%H-%M-%S")}'


def parse_logdir(s: str):
    return add_datetime(s) + "_" + platform.node()


custom_parsers = dict(logdir=parse_logdir)


def tree_dict_to_flat_dict(kv: Dict[str, Any]) -> Dict[str, Any]:
    # config_dict의 계층구조 제거
    df = pd.json_normalize(kv, sep="_")
    kv = df.to_dict(orient="records")
    kv = kv[0] if len(kv) == 1 else {}
    kv = {k.replace(".", "_"): v for k, v in kv.items()}
    return kv


class ConfigLoader:
    def __init__(self) -> None:
        self._config_items = []
        self.root = namedtuple("Config", "none")(None)
        self._revision = 0

    @singledispatchmethod
    def update(self, config_item: Union[str, PathLike, Dict[str, Any]]):
        raise NotImplementedError()

    @update.register
    def _update_config_as_str(self, config_item: str):
        config_file = Path(config_item)
        self._config_items.append(config_file)
        return self

    @update.register
    def _update_config_as_path(self, config_item: PathLike):
        self._config_items.append(config_item)
        return self

    @update.register
    def _update_config_as_dict(self, config_item: dict):
        self._config_items.append(tree_dict_to_flat_dict(config_item))
        return self

    def parse(self):
        main_file_path = Path(sys.argv[0])
        main_file_dir = main_file_path.parent
        main_file_name = main_file_path.name
        if main_file_path.name.endswith(".py"):
            main_file_name = main_file_path.name[:-3]

        # 초기 config_dict 설정
        init_config_dict = dict(
            verbose=1, command=" ".join(sys.argv), logdir=parse_logdir(main_file_name),
        )

        default_config_file = main_file_dir / f"{main_file_name}.toml"

        self.root, self._revision = self.load_config(
            [init_config_dict] + self._config_items + [default_config_file]
        )
        configure_logger(self.root.verbose)
        return self

    @staticmethod
    def load_config(config_items: Union[List[Path], Dict[str, Any]]):
        """
        config file, updated_dict, args_dict 순서대로 적용하여 config 객체 생성
        config file과 updated dict는 새로운 키를 추가할 수 있지만, 
        args_dict는 새로운 키를 추가 할 수 없음

        config_items에는 config_file과 config_dict가 list에 들어있으며,
        순서대로 파싱해서 이전 값을 업데이트 하므로, 앞에 있는 config 값보다 뒤에 있는 config 값의 우선순위가 높다.
        
        1. 코드에서 추가한 config_ile/dict (가장 우선순위 낮음)
        2. 실행 코드의 기본 config 파일
        3. argv에서 추가한 config file
        4. argv에서 추가한 config 값 (가장 우선순위 높음)
        
    
        Args:
            config_files (List[Path]): [description]
            updated_dict (Dict[str, Any]): [description]

        Returns:
            [type]: [description]
        """
        # argv 에서 config 검색해서 추가
        ARGV = []
        for s in copy.deepcopy(sys.argv):
            if s.startswith("--"):
                s = s[2:]
            ARGV.append(s)

        prefix = "config="
        new_config_files = [
            Path(kv[len(prefix) :]) for kv in ARGV[1:] if kv.startswith(prefix)
        ]
        for cf in new_config_files:
            if cf not in config_items:
                config_items

        # config load 및 병합
        # config items으로 기본 config 업데이트
        config_dict = {}
        for config_item in config_items:
            if isinstance(config_item, PathLike):
                # config file읽어 업데이트
                if config_item.exists():
                    kv = toml.loads(config_item.read_text())
                    # 계층구조 제거
                    kv = tree_dict_to_flat_dict(kv)
                    logger.info(f"{config_item} 읽기 성공")
                    for key, value in kv.items():
                        orig_value = config_dict.get(key)
                        config_dict[key] = value
                        logger.debug(f"{key}={orig_value} -> {value} 교체")

                elif config_item == config_items[-1]:
                    # 기본 config file은 없어도 에러 없이 무시
                    pass
                else:
                    logger.error(f"{config_item} 읽기 실패")
                    raise FileNotFoundError(config_item)

            elif isinstance(config_item, dict):
                for key, value in config_item.items():
                    orig_value = config_dict.get(key)
                    assert orig_value is None or type(orig_value) == type(value)
                    if config_dict.get(key) != config_item[key]:
                        logger.debug(f"{key}={orig_value} -> {value} 교체")
                    config_dict[key] = value

            else:
                raise NotImplementedError()

        #
        #  args_dict 생성후 config_dict 업데이트
        #

        # config 제외한 나머지 argv로 임시 설정 생성
        args_list = [kv.split("=") for kv in ARGV[1:] if not kv.startswith(prefix)]
        args_dict = {k.replace(".", "_"): v for k, v in args_list}

        # customize parser 적용
        for name, parser in custom_parsers.items():
            if name in args_dict:
                args_dict[name] = parser(args_dict[name])

        for key in args_dict:
            if key in config_dict:
                # command line으로 입력한 설정 중에서
                # config_dict + custom config에 사전에 정의어 있는 설정이 있으면
                # command line으로 입력한 값으로 업데이트
                try:
                    # key=value를 toml로 decoding
                    value_str = args_dict[key]
                    old_value = config_dict[key]
                    if isinstance(old_value, str):
                        new_args_dict = toml.loads(f'{key}="{value_str}"')
                    else:
                        new_args_dict = toml.loads(f"{key}={value_str}")

                    new_value = new_args_dict[key]
                    assert type(old_value) == type(new_value), "기존 값과 type이 같아야 함"

                except toml.TomlDecodeError:
                    logger.error(f"{key}={config_dict[key]} 디코딩 실패, toml 형식에 맞춰야함")
                    exit(1)

                old_value = config_dict[key]
                if old_value != new_value:
                    logger.debug(f"{key}={config_dict[key]} -> {new_value} 교체")
                    config_dict.update(new_args_dict)

            else:
                # 기존 config_dict에도 설정이 없고, argparse로 파싱 실패한 설정이 있으면 에러 발생

                # NOTE: command line입력으로는 기존 config를 수정할 수 만 있고, 새로 추가는 불가능
                # NOTE: 새로 추가가능하도록 허용하면, 기존 config가 업데이트되지 않아
                #   업데이트한 설정이 적용되지 않고, 에러도 발생하지 않기 때문에, 오작동해도 문제를
                #   발견하기 어렵기 때문에, 수정만 가능하도록 제한함
                logger.error(f"기존 config에 {key}가 없음, config에 있는 값만 변경가능")
                assert key in config_dict

        #
        #  수정 불가능하도록 읽기전용(tuple)으로 변경
        #
        for k in config_dict:
            if isinstance(config_dict[k], list):
                config_dict[k] = tuple(config_dict[k])

        #  root config 생성
        root_config = namedtuple("Config", list(config_dict.keys()))(**config_dict)
        for k, v in root_config._asdict().items():
            logger.debug(f"config> {k}={v}: ({type(v).__name__})")

        return root_config, len(config_items)

    def get(self, prefix: Optional[str] = None, clear_cache=False):
        if clear_cache:
            self._get.cache_clear()
        return self._get(prefix, revision=self._revision)

    @lru_cache
    def _get(self, prefix: Optional[str], revision: int):
        # 계층구조를 사용하지 않은 대신, flat한 key를 사용하고,
        # 일부 설정만 선택해서 사용하기 위해 키의 prefix로 필터링해서 필요한 설정만 선택

        if prefix is None:
            # 필터링 조건이 없으면 모든 설정을 반환
            max_len = max(len(k) for k in self.root._asdict())
            for k, v in self.root._asdict().items():
                logger.info(
                    f"config R{revision} {k:>{max_len}s} = {v}: ({type(v).__name__})"
                )
            return self.root

        prefix = prefix.replace(".", "_")  # .notation
        prefix_ = f"{prefix}_"

        new_cfg = {}
        for k, v in self.root._asdict().items():
            if k.startswith(prefix_):
                new_k = k[len(prefix_) :]
                new_cfg[new_k] = v

        if len(new_cfg) <= 0:
            logger.error(f"{prefix}로 시작하는 키가 없음")

        title = f"{prefix.replace('_', '').title()}Config"
        return namedtuple(title, list(new_cfg.keys()))(**new_cfg)


if __name__ == "__main__":

    # configure_logger(verbose_level=1)

    config_loader = ConfigLoader()
    # config_loader.add("test.toml")
    config_loader.update(dict(test="updated"))
    config_loader.parse()
    cfg = config_loader.get()

    exit()
