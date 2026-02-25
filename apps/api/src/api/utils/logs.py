from typing import Optional

from api.core.config import config
from logging import (
    basicConfig,
    Logger,
    getLogger,
    INFO,
    DEBUG,
    WARNING,
    ERROR,
)


class Log:
    def __init__(self):
        self.__logger: Optional[Logger] = None
        basicConfig(
            level=self.__logging_config(),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def __logging_config(self) -> int:
        if config.LOG_LEVEL == "INFO":
            return INFO
        elif config.LOG_LEVEL == "DEBUG":
            return DEBUG
        elif config.LOG_LEVEL == "WARNING":
            return WARNING
        elif config.LOG_LEVEL == "ERROR":
            return ERROR
        else:
            return INFO

    def get(self, filename: str) -> Logger:
        """Get a logger instance for the given filename."""
        if self.__logger is None:
            self.__logger = getLogger(filename)
        return self.__logger


logger: Logger = Log().get("api")
