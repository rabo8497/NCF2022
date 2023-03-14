import logging

from termcolor import colored


class ErrorLogException(Exception):
    pass


def configure_logger(n_verboses: int):
    #
    # error 호출하면 강제종료되도록 수정
    #
    orig_get_logger = logging.getLogger

    def get_logger(name: str):
        logger = orig_get_logger(name)

        orig_error_method = logger.error

        def error_method(msg: str):
            orig_error_method(msg)
            raise ErrorLogException(msg)

        logger.error = error_method

        return logger

    logging.getLogger = get_logger

    logger = logging.getLogger(__name__)

    #
    # root handler에 color 추가
    #
    logging.basicConfig(
        level=max(0, 30 - (n_verboses * 10)),
        format="%(levelname)s %(name)s:%(lineno)d %(message)s",
    )

    logger = logging.getLogger(__name__)

    _format = logger.root.handlers[0].formatter.format

    level_colors = {
        "INFO": "green",
        "DEBUG": "blue",
        "WARNING": "yellow",
    }

    def format(records):
        records.levelname = colored(
            records.levelname, level_colors.get(records.levelname, "red")
        )
        s = _format(records)
        return s

    logger.root.handlers[0].formatter.format = format
