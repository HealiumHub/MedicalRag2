import colorlog

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "()": colorlog.ColoredFormatter,
            "format": "{log_color}{asctime} [{levelname:5}] {filename}:{lineno:<5}- {funcName:<15}: {message}",
            "style": "{",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        },
    },
    "handlers": {
        # Outputs to file
        "file": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": "django.log",
        },
        # Output to console
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        # Using console for now.
        "handlers": ["console"],
        "level": "DEBUG",
    },
}
