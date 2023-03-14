import os

import psutil


def kill_all_processes(pid=None):
    if pid is None:
        pid = os.getpid()

    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()

