from enum import Enum


class TypeOfLayer(Enum):
    INPUT = 1,
    HIDDEN = 2,
    OUTPUT = 3
    UNDEFINED =4

    @classmethod
    def is_hidden(self, enum_value):
        if enum_value == TypeOfLayer.HIDDEN:
            return True
        return False