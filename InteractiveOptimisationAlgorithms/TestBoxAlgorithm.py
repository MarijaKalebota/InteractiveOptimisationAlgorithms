import numpy as np
from Matrix import *
from Functions import *
from Algorithms import *
from Drawing import *
from Constraints import *

EPSILON = 1E-6
ALPHA = 1
BETA = 0.5
GAMMA = 2
SIGMA = 0.5
PRINT = True

def main():
    f11 = F1RosenbrockBananaFunction()
    f12 = F1RosenbrockBananaFunction()
    f21 = F2()
    f22 = F2()
    elements1 = np.array([[-1.9, 2]])
    tocka11 =  Matrix(1,2,elements1)
    tocka12 = Matrix.copyPoint(tocka11)

    elements2 = np.array([[0.1, 0.3]])
    tocka21 =  Matrix(1,2,elements2)
    tocka22 =  Matrix.copyPoint(tocka21)

    #double[] donjeGranice = new double [] {-100,-100};
    donjeGranice = [-100, -100]
    gornjeGranice = [100, 100]
    impOgr1 = InequalityImplicitConstraint1()
    impOgr2 = InequalityImplicitConstraint2()

    implicitnaOgranicenja = [impOgr1,impOgr2]

    box_algorithm = BoxAlgorithm(f11, donjeGranice, gornjeGranice, implicitnaOgranicenja, EPSILON, ALPHA, PRINT)
    box_algorithm2 = BoxAlgorithm(f21, donjeGranice, gornjeGranice, implicitnaOgranicenja, EPSILON, ALPHA, PRINT)

    rjesenjePoBoxu1, logger1 = box_algorithm.run(tocka11)
    #rjesenjePoBoxu2 = BoxAlgorithm.run(f21,tocka21,donjeGranice.clone(),gornjeGranice.clone(),implicitnaOgranicenja,EPSILON,ALPHA,PRINT)
    rjesenjePoBoxu2, logger2 = box_algorithm2.run(tocka21)

    print "\n\n"
    print "Rjesenje po Boxu za f1: "
    Matrix.printMatrix(rjesenjePoBoxu1)
    print "Rjesenje po Boxu za f2: "
    Matrix.printMatrix(rjesenjePoBoxu2)

if (__name__ == '__main__'):
    main()