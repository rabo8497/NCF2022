import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from toolbox.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

#
# Team 저장소 설정
#

team_repo_url = {
    ## 예제 ##
    "Example10": "https://github.com/rex8312/NCF2022",
    ## 참가자 ##

}


#
# 기본 설정
#

config_loader = ConfigLoader()
config_loader.update(
    dict(
        # 평가 정보가 저장된 경로
        temp_dir=Path("__temp__"),
        out_dir=Path("eval_results"),
        ncf_repo="https://github.com/rex8312/NCF2022.git",
        # 작업 목록
        update_team_repo=True,
        play_games=True,
        export_results=True,
        publish_results=True,
        # 게임 플레이 옵션
        runs=100,
        timeout=1800,
        verbose=1,  # INFO까지 출력
        env="NetHackChallenge-v0",
    )
)
config_loader.parse()
args = config_loader.get()

verbose = args.verbose

# 평가시작시간
start_time = datetime.now()

# 임시폴더 및 결과 저장 경로
log_id = Path(f"{start_time.isoformat().replace(':', '-').split('.')[0]}")
temp_dir = args.temp_dir / log_id
temp_dir.mkdir(parents=True, exist_ok=True)
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

# 평가용 에이전트와 결과가 저장되는 폴더
agent_path_file = "agent_path.yaml"

team_dir = temp_dir / "team"
data_dir = temp_dir / "data"
system_log_file = temp_dir / "system.log"
result_file = temp_dir / "result.csv"

fig_dir = out_dir / "fig"
fig_dir.mkdir(parents=True, exist_ok=True)
summary_dir = out_dir / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "result.csv"


# 팀 정보 저장
@dataclass
class Team:
    name: str
    repo: str
    user_id: str
    repo_name: str
    temp_dir: str
    class_path: str


teams = dict()
for name, repo_url in team_repo_url.items():
    user_id, repo_name = repo_url.split("https://github.com/")[1].split("/")
    team_temp_dir = team_dir / user_id
    teams[name] = Team(name, repo_url, user_id, repo_name, team_temp_dir, None)
    del user_id
    del repo_name
    del team_temp_dir

# 게임결과 저장/읽기에 사용
csv_columns = [
    "agent",
    "run",
    "score",
    "play_time",
    "dates",
]

runs = args.runs

logger.debug("config 설정 종료")