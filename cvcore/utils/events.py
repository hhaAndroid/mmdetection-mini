__all__ = ['LoggerStorage', 'get_log_storage']

_CURRENT_LOGGER_STORAGE_STACK = []

def get_log_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _CURRENT_LOGGER_STORAGE_STACK
    ), "get_log_storage() has to be called inside a 'with LoggerStorage(...)' context!"
    return _CURRENT_LOGGER_STORAGE_STACK[-1]


class LoggerStorage:
    def __init__(self):
        self._storages = []

    def append(self, dict_obj):
        assert isinstance(dict_obj, dict)
        self._storages.append(dict_obj)

    def insert(self, index, dict_obj):
        assert isinstance(dict_obj, dict)
        self._storages.insert(index, dict_obj)

    def clear(self):
        self._storages = []

    def values(self):
        return self._storages

    def __enter__(self):
        _CURRENT_LOGGER_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_LOGGER_STORAGE_STACK[-1] == self
        _CURRENT_LOGGER_STORAGE_STACK.pop()
