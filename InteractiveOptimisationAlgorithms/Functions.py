import numpy as np
from Matrix import *

class IFunction(object):

    def valueAt(self, *args):
        '''
        if isinstance(args[0], np.float64):
            self.__valueAtPoint(*args)
        elif isinstance(args[0], Matrix):
            self.__valueAtMatrix(*args)
        '''
        raise NotImplementedError

    def __valueAtPoint(self, point):
        raise NotImplementedError

    def __valueAtMatrix(self, point, t=None):
        raise NotImplementedError

    def valueAtDerivativeByX1(self, point):
        raise NotImplementedError

    def valueAtDerivativeByX2(self, point):
        raise NotImplementedError

    def valueAtDerivativeByX1ThenX1(self, point):
        raise NotImplementedError

    def valueAtDerivativeByX1ThenX2(self, point):
        raise NotImplementedError

    def valueAtDerivativeByX2ThenX1(self, point):
        raise NotImplementedError

    def getNumberOfCalls(self):
        raise NotImplementedError

class IAlgorithm(object):

    def run(self, initialPoint):
        raise NotImplementedError

class AbstractFunction(IFunction):

    def __init__(self):
        self.__counter = 0

    def increment(self):
        self.__counter =+ 1

    def getNumberOfCalls(self):
        return self.__counter



class F3OneDimensional(AbstractFunction):

    '''public F3OneDimensional() {
        super();
    }
    '''

    def __init__(self):
        #super.__init__()
        AbstractFunction.__init__(self)

    '''
    def valueAt(self, point):
        #super.increment()
        AbstractFunction.increment()
        #return Math.pow(point - 3, 2);
        return (point - 3)**2
    '''

    def valueAt(self, *args):
        # if isinstance(args[0], np.float64):
        #     return self.__valueAtPoint(*args)
        # elif isinstance(args[0], int):
        #     return self.__valueAtPoint(*args)
        # elif isinstance(args[0], Matrix):
        #     return self.__valueAtMatrix(*args)
        if isinstance(args[0], Matrix):
            #increment()
            return self.__valueAtMatrix(*args)
        else:
            return self.__valueAtPoint(*args)

    def __valueAtPoint(self, point):
        AbstractFunction.increment(self)
        return (point - 3) ** 2

    def __valueAtMatrix(self, point):
        if(point.getColsCount()!=1 and point.getRowsCount()!=1):
            raise ValueError("Tocka mora biti u jednodimenzijskom prostoru.")
        return F3OneDimensional.__valueAtPoint(self, point.getElement(0, 0))

    '''
    def valueAt(self, point, t):
        return 0
    '''

    def valueAtDerivativeByX1(self, point):
        return 0

    def valueAtDerivativeByX2(self, point):
        return 0

    def valueAtDerivativeByX1ThenX1(self, point):
        return 0

    def valueAtDerivativeByX1ThenX2(self, point):
        return 0

    def valueAtDerivativeByX2ThenX1(self, point):
        return 0

    def valueAtDerivativeByX2ThenX2(self, point):
        return 0

class F1RosenbrockBananaFunction(AbstractFunction):
    def __init__(self):
        #super.__init__()
        AbstractFunction.__init__(self)

    def valueAt(self, *args):
        if isinstance(args[0], Matrix):
            return self.__valueAtMatrix(*args)
        else:
            print "Poslani argument mora biti tipa Matrix"

    def __valueAtMatrix(self, point):
        if (point.getColsCount() != 2):
            print "Funckija prima dva parametra."
        AbstractFunction.increment(self)
        return 100 * (point.getElement(0, 1) - point.getElement(0, 0)**2)**2 + (1 - point.getElement(0, 0))**2



    def valueAtDerivativeByX1(self, point):
        return (-400) * (point.getElement(0, 1) - point.getElement(0, 0)**2) * point.getElement(0,0) + 2 * point.getElement(0, 0) - 2


    def valueAtDerivativeByX2(self, point):
        return 200 * (point.getElement(0, 1) - point.getElement(0, 0)**2)


    def valueAtDerivativeByX1ThenX1(self, point):
        #return (800 * Math.pow(point.getElement(0, 0), 2) - 2);
        return 1200 * point.getElement(0, 0)**2 - 400 * point.getElement(0, 1) + 2


    def valueAtDerivativeByX1ThenX2(self, point):
        return -400 * point.getElement(0, 0)


    def valueAtDerivativeByX2ThenX1(self, point):
        return -400 * point.getElement(0, 0)


    def valueAtDerivativeByX2ThenX2(self, point):
        return 200

class F4xcosx(AbstractFunction):

    '''public F3OneDimensional() {
        super();
    }
    '''

    def __init__(self):
        # super.__init__()
        AbstractFunction.__init__(self)


    def valueAt(self, *args):
        # if isinstance(args[0], np.float64):
        #     return self.__valueAtPoint(*args)
        # elif isinstance(args[0], int):
        #     return self.__valueAtPoint(*args)
        # elif isinstance(args[0], Matrix):
        #     return self.__valueAtMatrix(*args)
        if isinstance(args[0], Matrix):
            # increment()
            return self.__valueAtMatrix(*args)
        else:
            return self.__valueAtPoint(*args)

    def __valueAtPoint(self, point):
        AbstractFunction.increment(self)
        return point * np.cos(point)

    def __valueAtMatrix(self, point):
        if (point.getColsCount() != 1 and point.getRowsCount() != 1):
            raise ValueError("Tocka mora biti u jednodimenzijskom prostoru.")
        return F4xcosx.__valueAtPoint(self, point.getElement(0, 0))

    '''
    def valueAt(self, point, t):
        return 0
    '''

    def valueAtDerivativeByX1(self, point):
        return 0

    def valueAtDerivativeByX2(self, point):
        return 0

    def valueAtDerivativeByX1ThenX1(self, point):
        return 0

    def valueAtDerivativeByX1ThenX2(self, point):
        return 0

    def valueAtDerivativeByX2ThenX1(self, point):
        return 0

    def valueAtDerivativeByX2ThenX2(self, point):
        return 0
