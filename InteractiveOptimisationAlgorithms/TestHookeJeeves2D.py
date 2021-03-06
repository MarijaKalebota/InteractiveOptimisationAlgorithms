import numpy as np
#import Matrix
from Matrix import *
#import Functions
from Functions import *
from Algorithms import *
from Drawing import *


f1 = F1RosenbrockBananaFunction()
elements1 = np.array([[-1.9, 2]])
point1 =  Matrix(1, 2, elements1)
h_j_algorithm = HookeJeeves(f1, 1, 0.5, 1E-6, True)
#solutionHookeJeeves1, logger = HookeJeeves.run(f1, point1, 1, 0.5, 1E-6, True)
solutionHookeJeeves1, logger = h_j_algorithm.run(point1)