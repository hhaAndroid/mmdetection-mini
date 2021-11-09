import warnings

__all__ = ['DefaultFuncStorage', 'get_func_storage']

_CURRENT_FUNC_STORAGE_STACK = []


def get_func_storage():
    if len(_CURRENT_FUNC_STORAGE_STACK) == 0:
        warnings.warn("get_func_storage() has to be called inside a 'with DefaultFuncStorage(...)' context! "
                      "If you do not set `runtime_func` in config, skip directly by default")
        return NoErrorClass()
    else:
        return _CURRENT_FUNC_STORAGE_STACK[-1]


class NoErrorClass:
    def __getattr__(self, item):
        def hacky_func(*args, **kwargs):
            pass
        return hacky_func


class DefaultFuncStorage:
    def __enter__(self):
        _CURRENT_FUNC_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_FUNC_STORAGE_STACK[-1] == self
        _CURRENT_FUNC_STORAGE_STACK.pop()
