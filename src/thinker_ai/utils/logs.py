import sys

from loguru import logger as _logger

from thinker_ai.config import get_project_root


def define_log_level(log_dir:str,print_level="INFO", logfile_level="DEBUG"):
    """调整日志级别到level之上
       Adjust the log level to above level
    """
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(f'{log_dir}/log.txt', level=logfile_level)
    return _logger


logger = define_log_level(str(get_project_root()))
