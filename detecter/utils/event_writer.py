__all__ = ['EventWriter']


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """
    def init(self, runner):
        pass

    def write(self):
        raise NotImplementedError

    def close(self):
        pass
