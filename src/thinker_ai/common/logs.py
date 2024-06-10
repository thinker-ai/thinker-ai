import sys

from loguru import logger as _logger

from thinker_ai.configs.const import PROJECT_ROOT

_print_level = "INFO"

def define_log_level(log_dir:str,print_level="INFO", logfile_level="DEBUG"):
    """调整日志级别到level之上
       Adjust the log level to above level
    """
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(f'{log_dir}/log.txt', level=logfile_level)
    return _logger

logger = define_log_level(PROJECT_ROOT)


def log_llm_stream(msg):
    _llm_stream_log(msg)


def set_llm_stream_logfunc(func):
    global _llm_stream_log
    _llm_stream_log = func


def _llm_stream_log(msg):
    if _print_level in ["INFO"]:
        print(msg, end="")

