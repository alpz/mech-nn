import logging
from logging import handlers
#from logging.handlers import RotatingFileHandler
#from logging.handlers import FileHandler
import sys
import os


def setup(path, stdout=True):
    #logging.basicConfig(
    #    level=logging.INFO,
    #    format="%(message)s",
    #    handlers=[
    #        logging.FileHandler(path+"/debug.log"),
    #        logging.StreamHandler(sys.stdout)
    #    ]
    #)
    #main_log.setLevel(logging.DEBUG)
    main_log = logging.getLogger('')
    main_log.setLevel(logging.INFO)
    format = logging.Formatter("%(message)s")

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        main_log.addHandler(ch)

    #fh = handlers.RotatingFileHandler(LOGFILE, maxBytes=(1048576*5), backupCount=7)
    file_format = logging.Formatter("%(asctime)s - %(message)s")

    log_path = os.path.join(path, 'log.txt')
    #fh = handlers.FileHandler(log_path)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(file_format)
    main_log.addHandler(fh)


    return main_log
