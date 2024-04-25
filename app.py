import logging
import logging.config

from config import LOGGING_CONFIG
from ui.app_controller import AppController

logging.config.dictConfig(LOGGING_CONFIG)

AppController()
