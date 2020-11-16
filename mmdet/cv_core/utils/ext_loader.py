import importlib
import pkgutil


def load_ext(name, funcs):
    # 考虑到很多人无法编译成功，故直接替换为mmcv已经编译版本
    # ext = importlib.import_module('mmdet.cv_core.' + name)
    ext = importlib.import_module('mmcv.' + name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext


def check_ops_exist():
    # ext_loader = pkgutil.find_loader('mmdet.cv_core._ext')
    ext_loader = pkgutil.find_loader('mmcv')
    return ext_loader is not None
