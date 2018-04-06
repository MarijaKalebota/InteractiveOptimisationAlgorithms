#from Matrix import *

class IConstraint(object):

    def is_satisfied(self, point):
        raise NotImplementedError

    def value_at(self, point):
        raise NotImplementedError

    def get_gradient(self):
        raise NotImplementedError

class ExplicitConstraint(IConstraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def is_satisfied(self, point):
        raise NotImplementedError
    def value_at(self, point):
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
        return 2 - point.getElement(0,0)

class EqualityImplicitConstraint3(IConstraint):
    def is_satisfied(self, point):
        if (point.getElement(0, 1) - point.getElement(0, 0) + 1 == 0):
            return True
        else:
            return False

    def value_at(self, point):
        return point.getElement(0, 1) - point.getElement(0, 0) + 1