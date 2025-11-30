import os
import logging
import coloredlogs
from tqdm.auto import tqdm
from pathlib import Path
import shutil


DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
if shutil.which("sbatch"):
    SCHEDULER = "slurm"
elif shutil.which("qsub"):
    SCHEDULER = "pbs"
else:
    SCHEDULER = "local"

args = {
    "data_root": Path("data/360-1M/"),
    "debug_root": Path("debug"),
    "stella_vslam_path": Path("~/lib/stella_vslam_examples/build/run_video_slam").expanduser(),
    "fbow_path": Path("~/lib/stella_vslam_examples/build/orb_vocab.fbow").expanduser(),
}


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(
    name,
    output_path=None,
    colored=True,
):
    logger = logging.getLogger(name)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(LOGGING_LEVEL)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = TqdmLoggingHandler()
    handler.setLevel(LOGGING_LEVEL)
    logger.addHandler(handler)

    log_format = "%(asctime)s %(funcName)s %(name)s [%(levelname)s]: %(message)s"

    if colored:
        log_styles = {
            "debug": {"color": "white"},
            "info": {"color": "blue"},
            "warning": {"color": "yellow"},
            "error": {"color": "red"},
            "critical": {"color": "magenta", "bold": True},
        }
        coloredlogs.install(
            level=LOGGING_LEVEL,
            logger=logger,
            fmt=log_format,
            level_styles=log_styles,
            isatty=True,
        )
    else:
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

    if output_path:
        log_path = Path(output_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        # file_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(LOGGING_LEVEL)
        logger.addHandler(file_handler)

    return logger
