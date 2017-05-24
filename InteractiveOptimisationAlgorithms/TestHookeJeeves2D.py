import numpy as np
#import Matrix
from Matrix import *
#import Functions
from Functions import *
from Test0 import HookeJeeves


f1 = F1RosenbrockBananaFunction()
elements1 = np.array([[-1.9, 2]])
tocka1 =  Matrix(1,2,elements1)
rjesenjeHookeJeeves1, logger = HookeJeeves.run(f1, tocka1, 1, 0.5, 1E-6, True)