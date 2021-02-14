import os
import logging

logger = logging.getLogger(__name__)

def set_root_logger():
    """Configures root logger. This function may only be called once to avoid duplicate logging output

    :return: None
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s; %(name)s; %(levelname)s; %(message)s')
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    logger.info('Root logger is set up')

def make_sure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f'Created directory ``{dir_path}``')