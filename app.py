import logging.config
from ui.app_controller import AppController
import logging
from config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

AppController()
