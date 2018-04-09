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

    def is_satisfied(self, value):
        if(value > self.upper_bound or value < self.lower_bound):
            return False
        else:
            return True

    def value_at(self, point):
        raise NotImplementedError

class InequalityImplicitConstraint1(IConstraint):
    def is_satisfied(self, point):
        if(point.getElement(0,1) - point.getElement(0,0) >= 0):
            return True
        else:
            return False

    def value_at(self, point):
        return point.getElement(0,1) - point.getElement(0,0)

class InequalityImplicitConstraint2(IConstraint):
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

class EqualityImplicitConstraint4(IConstraint):
    def is_satisfied(self, point):
        if (point.getElement(0, 1) - point.getElement(0, 9) + 50 == 0):
            return True
        else:
            return False

    def value_at(self, point):
        return point.getElement(0, 1) - point.getElement(0, 0) + 50