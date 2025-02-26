import logging
import sys

file_logger = logging.getLogger("playpen_examples")
file_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("log.log", encoding="utf-8")
formatter = logging.Formatter("{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M")
file_handler.setFormatter(formatter)
file_logger.addHandler(file_handler)


out_logger = logging.getLogger("playpen_examples_out")
out_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
out_logger.addHandler(stream_handler)