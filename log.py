import logging


def __loggerConfig(loggerPath: str, loggerName: str):
    logger = logging.getLogger(loggerName)
    logger.setLevel(level=logging.DEBUG)
    filter = logging.Filter("radiomicsTool")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s"
    )
    handler = logging.FileHandler(loggerPath, encoding="UTF-8")
    handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    handler.addFilter(filter)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


logger = __loggerConfig(loggerPath="./radiomicstool.log",
                        loggerName="radiomicsTool")
