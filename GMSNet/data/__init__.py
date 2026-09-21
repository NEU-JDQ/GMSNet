"""Audit metadata without importing the ML dependencies."""
from importlib import import_module
from .config import TASK_CONFIG, DEFAULT_DATA_ROOT

__all__ = ['CrisisDataset', 'TextProcessor', 'get_transforms',
           'TASK_CONFIG', 'DEFAULT_DATA_ROOT']


def __getattr__(name):
    modules = {'CrisisDataset': '.dataset', 'TextProcessor': '.text_proc',
               'get_transforms': '.transforms'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
