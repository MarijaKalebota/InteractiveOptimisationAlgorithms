from Matrix import *

class IConstraint(object):

    def is_satisfied(self, point):
        raise NotImplementedError

    def value_at(self, point):
        raise NotImplementedError

    def get_gradient(self):
        raise NotImplementedError


class ImplicitConstraint1(IConstraint):
    def is_satisfied(self, point):
        if(point.getElement(0,1) - point.getElement(0,0) >= 0):
            return True
        else:
            return False

    def value_at(self, point):
        return point.getElement(0,1) - point.getElement(0,0)

class ImplicitConstraint2(IConstraint):
    def is_satisfied(self, point):
        if(2 - point.getElement(0,0) >= 0):
            return True
        else:
            return False

    def value_at(self, point):
        if(self.is_satisfied(point)):
            return 2 - point.getElement(0,0)
        else:
            return float('-inf')