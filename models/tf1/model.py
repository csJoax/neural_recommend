from typing import Union, Iterable
from abc import ABC, abstractmethod


# TODO 暂时不使用，只留下接口
class RecommendModel(ABC):
    """
    推荐模型
    """

    def __init__(self):
        self._input = None
        self._user = None
        self._item = None
        self._item = None
        self._item = None
        pass
