from Algorithms import *
def main():
    f3OneDimensional1 = F3OneDimensional()
    f3OneDimensional12 = F3OneDimensional()
    elements = np.array([[10]])
    point1 = Matrix(1, 1, elements)
    point12 = Matrix.copyPoint(point1)
    interval = GoldenSection.runGoldenSectionWithInitialPoint(point1.getElement(0, 0), 1E-6, f3OneDimensional1, True)
    h_j_algorithm = HookeJeeves(f3OneDimensional12, 1, 0.5, 1E-6, True)
    #solutionHookeJeeves, loggerHookeJeeves = HookeJeeves.provediHookeJeeves(f3OneDimensional12, point12, 1, 0.99, 1E-6, True)
    #solutionHookeJeeves, loggerHookeJeeves = HookeJeeves.run(f3OneDimensional12, point12, 1, 0.5, 1E-6, True)
    solutionHookeJeeves, loggerHookeJeeves = h_j_algorithm.run(point12)

    #print point1.getRowsCount()
    #Matrix.printMatrix(solutionHookeJeeves)

if (__name__ == '__main__'):
    main()