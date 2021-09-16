# Copyright (c) Facebook, Inc. and its affiliates.
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["PathManager", "PathHandler"]

PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())



