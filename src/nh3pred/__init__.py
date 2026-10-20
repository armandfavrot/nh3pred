from .prediction import predict

__all__ = ["predict"]

from importlib.metadata import version

__version__ = version("nh3pred")
