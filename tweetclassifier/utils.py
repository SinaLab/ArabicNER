import sys
import logging
import importlib


def logging_config(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, "w", "utf-8"))
        print("Logging to {}".format(log_file))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )


def load_object(name, kwargs):
    object_module, object_name = name.rsplit(".", 1)
    object_module = importlib.import_module(object_module)
    fn = getattr(object_module, object_name)(**kwargs)
    return fn
