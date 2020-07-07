from abc import ABC


class Registerable:
    _REGISTRY = None
    _ID = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        assert cls._REGISTRY is not None, f"No registry specified for class {cls}"
        if cls._ID is not None:
            assert (
                cls._ID not in cls._REGISTRY
            ), f"Duplicate key {cls._ID} in {cls._REGISTRY}"
            cls._REGISTRY[cls._ID] = cls

    @classmethod
    def get(cls, identifier):
        return cls._REGISTRY[identifier]

    @classmethod
    def list_identifiers(cls):
        return list(cls._REGISTRY.items())

    @classmethod
    def _doc(cls, *args):
        return cls.__doc__
