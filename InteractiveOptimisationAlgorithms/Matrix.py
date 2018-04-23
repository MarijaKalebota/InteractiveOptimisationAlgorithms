'''package fer.apr.DZ1;

import fer.apr.DZ2.HookeJeeves;

import java.io.*;
import java.util.*;

import java.util.Arrays;

/**
 * Created by Marija on 22.10.2016..
 */
 '''
import numpy as np

class Matrix:

    def __init__(self, rowNumber, columnNumber, elements):
        self.numberOfRows = rowNumber
        self.numberOfColumns = columnNumber
        self.elements = elements

    def getElements(self):
        return self.elements

    @staticmethod
    def printMatrix(a):
        #StringBuilder sb = new StringBuilder();
        #sb.append("[\n");
        outputString = "[\n"
        #for(int i = 0; i < a.numberOfRows; i ++){
        #print "a.getRowsCount() = " +str(a.getRowsCount())
        for i in range(a.getRowsCount()):
            outputString += "[ "
            for j in range(a.getColsCount()):
                #TODO upper or lower?
                # outputString += a.elements[i][j]
                outputString += str(a.elements[i,j])
                outputString += "\t"
            outputString += "]\n"
        outputString += "]\n"
        print outputString

    #pridruzivanje
    def makeEqualTo(self, B):
        self.numberOfColumns = B.getColsCount()
        self.numberOfRows = B.getRowsCount()
        self.elements = np.array(B.getElements())
        return B

    def equals(self, obj):
        if (self == obj):
            return True
        if (obj == None):
            return False
        #TODO
        #if (getClass() != obj.getClass())
            #return false;
        #other = (Matrix)obj
        if self.getColsCount() != obj.getColsCount():
            return False
        #TODO
        #if (!Arrays.deepEquals(elements, other.elements))
            #return false;
        if self.getRowsCount() != self.getRowsCount():
            return False
        return True

    def getElement(self, row, column):
        if row >= self.getRowsCount() or column >= self.getColsCount():
            #throw new IndexOutOfBoundsException("Given index is outside matrix bounds.");
            print "Given index is outside matrix bounds."
        if (row < 0 or column < 0):
            #throw new IndexOutOfBoundsException("Index needs to be a positive number.");
            print "Index needs to be a positive number."
        test = self.elements[row,column]
        return self.elements[row,column]

    def setElement(self, row, column, value):
        if (row >= self.numberOfRows or column >= self.numberOfColumns):
            # throw new IndexOutOfBoundsException("Given index is outside matrix bounds.");
            print "Given index is outside matrix bounds."
        if (row < 0 or column < 0):
            # throw new IndexOutOfBoundsException("Index needs to be a positive number.");
            print "Index needs to be a positive number."
        self.elements[row, column] = value
        return self

    def getRowsCount(self):
        return self.numberOfRows

    def getColsCount(self):
        return self.numberOfColumns

    @staticmethod
    #TODO assert a and b?
    def add(a, b):
        if(a.numberOfColumns != b.numberOfColumns or a.numberOfRows != b.numberOfRows):
            print "Matrices are of different dimensions and cannot be added."
            return None

        #double [][] resultElements = new double[a.numberOfRows][a.numberOfColumns];
        resultElements = np.empty([a.numberOfRows, a.numberOfColumns], dtype=np.float64)
        for i in range(a.numberOfRows):
            for j in range(a.numberOfColumns):
                #TODO upper or lower line?
                #resultElements[i][j] = a.elements[i][j] + b.elements[i][j]
                resultElements[i,j] = a.elements[i,j] + b.elements[i,j]
        resultMatrix = Matrix(a.numberOfRows, a.numberOfColumns, resultElements)
        return resultMatrix

    @staticmethod
    def subtract(a, b):
        if(a.numberOfColumns != b.numberOfColumns or a.numberOfRows != b.numberOfRows):
            print "Matrices are of different dimensions, subtraction cannot be done."
            return None

        resultElements = np.empty([a.numberOfRows,a.numberOfColumns], dtype = np.float64)
        for i in range(a.numberOfRows):
            for j in range (a.numberOfColumns):
                #TODO upper or lower
                #resultElements[i][j] = a.elements[i][j] - b.elements[i][j];
                resultElements[i,j] = a.elements[i,j] - b.elements[i,j]
        resultMatrix = Matrix(a.numberOfRows, a.numberOfColumns, resultElements)
        return resultMatrix

    def plusEquals(self, matrix):
        #if (self.getColsCount() != matrix.getColsCount() || this.getRowsCount() != matrix.getRowsCount())
            #throw new IncompatibleOperandException("Matrices are of different dimensions.");
        #for (int i = this.getRowsCount() - 1; i >= 0; --i)
        for i in range(self.getRowsCount() - 1, -1, -1):
            #for (int j = this.getColsCount() - 1; j >= 0; --j)
            for j in range(self.getColsCount() - 1, -1, -1):
                self.setElement(i, j, self.getElement(i, j) + matrix.getElement(i, j))
        return self

    def minusEquals(self, matrix):
        #if (self.getColsCount() != matrix.getColsCount() || this.getRowsCount() != matrix.getRowsCount())
            #throw new IncompatibleOperandException("Matrices are of different dimensions.");
        #for (int i = this.getRowsCount() - 1; i >= 0; --i)
        for i in range(self.getRowsCount() - 1, -1, -1):
            #for (int j = this.getColsCount() - 1; j >= 0; --j)
            for j in range(self.getColsCount() - 1, -1, -1):
                self.setElement(i, j, self.getElement(i, j) - matrix.getElement(i, j))
        return self

    def multiply(self, matrix):
        #if (self.getColsCount() != matrix.getColsCount() || this.getRowsCount() != matrix.getRowsCount())
            #throw new IncompatibleOperandException("Matrices are of different dimensions.");
        #for (int i = this.getRowsCount() - 1; i >= 0; --i)
        elements = np.empty([self.getRowsCount()][matrix.getColsCount()], dtype=np.float64)
        for i in range(self.getRowsCount() - 1, -1, -1):
            #for (int j = this.getColsCount() - 1; j >= 0; --j)
            for j in range(matrix.getColsCount() - 1, -1, -1):
                for k in range(matrix.getRowsCount() - 1, -1, -1):
                    elements[i,j] += self.getElement(i,k) * matrix.getElement(k,j)
        return Matrix(self.getRowsCount(), matrix.getColsCount(), elements)

    @staticmethod
    def multiply(a, b):
        if(a.numberOfColumns != b.numberOfRows):
            print "Matrices are not compatible for multiplication."
            return None

        #double [][] resultElements = new double[a.numberOfRows][b.numberOfColumns];
        resultElements = np.empty([a.numberOfRows][b.numberOfColumns], dtype=np.float64)
        for i in range(a.numberOfRows):
            for j in range (b.numberOfColumns):
                resultElement = 0
                for k in range(a.numberOfColumns):
                    #TODO upper or lower?
                    # resultElement = resultElement + a.elements[i][k]*b.elements[k][j];
                    resultElement = resultElement + a.elements[i,k] * b.elements[k,j]
                #TODO
                resultElements[i,j] = resultElement
        resultMatrix = Matrix(a.numberOfRows, b.numberOfColumns, resultElements)
        return resultMatrix

    @staticmethod
    def scalarMultiply(matrix, scalar):
        #double[][] resultElements = new double[matrix.numberOfRows][matrix.numberOfColumns];
        resultElements = np.empty([matrix.numberOfRows,matrix.numberOfColumns], dtype=np.float64)
        for i in range(matrix.getRowsCount()):
            for j in range(matrix.getColsCount()):
                #TODO
                #resultElements[i][j] = matrix.elements[i][j] * scalar;
                resultElements[i,j] = matrix.elements[i,j] * scalar
        resultMatrix = Matrix(matrix.numberOfRows, matrix.numberOfColumns, resultElements)
        return resultMatrix

    @staticmethod
    def sanityCheck(d):
        EPSILON = 1E-6
        if((abs(round(d) - d) < EPSILON) and (abs(round(d) - d) != 0) ):
            #print "Pojavila se vrijednost " + str(d) + " koja ima decimalno odstupanje od cijelog broja manje od epsilon = " + str(EPSILON) + ". Postavljam na " + str(round(d))
            print "Rounding " + str(d) + " to " + str(round(d))
            return round(d)
        return d

    @staticmethod
    def copy_matrix(matrix):
        '''copy = np.empty([matrix.getRowsCount()][matrix.getColsCount()], dtype= np.float64)
        for i in range(matrix.getRowsCount()):
            copy[i]=np.array(matrix.getElements()[i])

        copyMatrix = Matrix(matrix.getRowsCount(), matrix.getColsCount(), copy)
        return copyMatrix
        '''
        return Matrix(matrix.getRowsCount(), matrix.getColsCount(), np.array(matrix.getElements()))

    @staticmethod
    def forwardSubstitution(a, b):
        if(b.numberOfRows!=a.numberOfRows):
            print "Vector b has different dimension than matrix - cannot do forward substitution."
        resultB = Matrix(b.numberOfRows, b.numberOfColumns, np.array(b.elements))
        #double[][] resultElements = new double[b.numberOfRows][1];
        resultElements = np.empty([b.numberOfRows][1], dtype=np.float64)

        for i in range(a.numberOfRows - 1):
            for j in range(i + 1, a.numberOfRows, 1):
                #TODO
                #resultB.elements[j][0] = sanityCheck(resultB.elements[j][0] - a.elements[j][i]*resultB.elements[i][0]);
                resultB.elements[j,0] = Matrix.sanityCheck(resultB.elements[j,0] - a.elements[j,i] * resultB.elements[i,0])
        return resultB

    @staticmethod
    def backwardSupstitution(a, vector):
        if(vector.numberOfRows!=a.numberOfRows):
            print "Vector y has different dimension than matrix - cannot do backward substitution."
        for i in range(a.numberOfRows-1, -1, -1):
            EPSILON = 1E-6
            if(abs(a.getElement(i, i))<EPSILON):
                print "Upper triangular matrix has zero on diagonal - cannot do backward substitution."
                return None
            vector.setElement(i, 0, Matrix.sanityCheck(vector.getElement(i, 0)/a.getElement(i,i)))
            for j in range(i):
                vector.setElement(j, 0, Matrix.sanityCheck(vector.getElement(j, 0)-a.getElement(j, i)*vector.getElement(i, 0)))
        #return new Matrica(1, vector.colSize(), vector.getElements());
        return vector

    @staticmethod
    def swapRowsOfMatrix(matrix, firstRow, secondRow):
        #TODO verify
        tmp = matrix.elements[firstRow,:]
        matrix.elements[firstRow] = matrix.elements[secondRow,:]
        matrix.elements[secondRow,:] = tmp

    @staticmethod
    def lupDecomposition(a, permutationMatrix, lup):
        if (a.numberOfRows != a.numberOfColumns):
            #throw new IllegalArgumentException("Matrix must be square.");
            print "Matrix must be square."
        identityMatrixElements = Matrix.createIdentityMatrixElements(a.numberOfRows)
        workingMatrix = Matrix.copy_matrix(a)
        Matrix.printMatrix(workingMatrix)
        for i in range(a.numberOfRows - 1):
            if (lup):
                #index = findMaxApsElement(workingMatrix, i, workingMatrix.elements[i][i])
                index = Matrix.findMaxApsElement(workingMatrix, i, workingMatrix.elements[i,i])
                if (index != i):
                    Matrix.swapRowsOfMatrix(permutationMatrix, i, index)
                    Matrix.swapRowsOfMatrix(workingMatrix, i, index)
                    print "Swapping rows " + str(i) + " and " + str(index)
                    Matrix.printMatrix(workingMatrix)
            pivot = workingMatrix.getElement(i, i)
            if (pivot == 0):
                Matrix.printMatrix(workingMatrix)
                print "Pivot is zero - cannot do decomposition"
                return None
            for j in range(i + 1, a.numberOfRows, 1):
                workingMatrix.setElement(j, i, Matrix.sanityCheck(workingMatrix.getElement(j, i) / float(pivot)))
                for k in range(i + 1,a.numberOfRows, 1):
                    workingMatrix.setElement(j, k, Matrix.sanityCheck(workingMatrix.getElement(j, k) - workingMatrix.getElement(j, i) * workingMatrix.getElement(i, k)))
            Matrix.printMatrix(workingMatrix);
        #TODO maybe return both working matrix and permutation matrix?
        #print "Vracam dekomponiranu matricu, a predana permutacijska matrica promijenjena je ako je bilo potrebno. Njome sada treba pomnoziti vektor b."
        return workingMatrix

    @staticmethod
    def createIdentityMatrixElements(n):
        elements = np.zeros((n,n), dtype=np.float64)
        for i in range(n):
            #TODO
            #elements[i][i] = 1;
            elements[i,i] = 1
        return elements

    @staticmethod
    def findMaxApsElement(matrix, startIndex, value):
        indexOfMax = startIndex
        max = value
        for i in range(startIndex, matrix.numberOfRows, 1):
            if (abs(matrix.getElement(i, startIndex)) > max):
                max = abs(matrix.getElement(i, startIndex))
                indexOfMax = i
        return indexOfMax

    @staticmethod
    def transpose(matrix):
        resultElements = np.empty([matrix.numberOfColumns,matrix.numberOfRows], dtype=np.float64)
        for i in range(matrix.numberOfRows):
            for j in range(matrix.numberOfColumns):
                #TODO
                #resultElements[j][i] = matrix.elements[i][j];
                resultElements[j,i] = matrix.elements[i,j]
        resultMatrix = Matrix(matrix.numberOfColumns, matrix.numberOfRows, resultElements)
        return resultMatrix

    @staticmethod
    def hasInverse(matrix):
        if(matrix.numberOfRows != matrix.numberOfColumns):
            print "Matrix is not square, does not have inverse"
            return False
        determinant = Matrix.determinant(matrix)
        if(determinant == 0):
            print "Matrix is singular, does not have inverse"
            return False
        return True

    @staticmethod
    def determinant(matrix):
        '''return matrix.elements[0][0] * matrix.elements[1][1] * matrix.elements[2][2] +
                matrix.elements[0][1] * matrix.elements[1][2] * matrix.elements[2][0] +
                matrix.elements[0][2] * matrix.elements[1][0] * matrix.elements[2][1] -
                matrix.elements[2][0] * matrix.elements[1][1] * matrix.elements[0][2] -
                matrix.elements[2][1] * matrix.elements[1][2] * matrix.elements[0][0] -
                matrix.elements[2][2] * matrix.elements[1][0] * matrix.elements[0][1];
        '''
        return matrix.elements[0,0] * matrix.elements[1,1] * matrix.elements[2,2] + matrix.elements[0,1] * matrix.elements[1,2] * matrix.elements[2,0] + matrix.elements[0,2] * matrix.elements[1,0] * matrix.elements[2,1] - matrix.elements[2,0] * matrix.elements[1,1] * matrix.elements[0,2] - matrix.elements[2,1] * matrix.elements[1,2] * matrix.elements[0,0] - matrix.elements[2,2] * matrix.elements[1,0] * matrix.elements[0,1]

    @staticmethod
    def invert(matrixToInvert):
        #Ax = prvi stupac jedinicne
        #=> x je prvi stupac invertirane, itd. za svaki stupac

        copyOfMatrixToInvert = Matrix.copy_matrix(matrixToInvert)
        identityMatrixElements = Matrix.createIdentityMatrixElements(copyOfMatrixToInvert.getRowsCount()) #create identity matrix;
        identityMatrix = Matrix(copyOfMatrixToInvert.getRowsCount(), copyOfMatrixToInvert.getColsCount(), identityMatrixElements)

        decomposed = Matrix.lupDecomposition(copyOfMatrixToInvert,identityMatrix,True)
        elementsOfFinalInvertedMatrix = np.empty([matrixToInvert.getColsCount(), matrixToInvert.getRowsCount()], dtype= np.float64)

        for j in range(matrixToInvert.getColsCount()):
            elementsOfY = np.empty([matrixToInvert.getRowsCount(), 1], dtype=np.float64)
            #TODO
            # elementsOfY[j][0] = 1;
            elementsOfY[j,0] = 1
            y = Matrix(matrixToInvert.getRowsCount(), 1, elementsOfY)
            permutatedY = Matrix.multiply(identityMatrix,y)
            forwardSubstitutedY = Matrix.forwardSubstitution(decomposed, permutatedY)
            backwardSubstitutedY = Matrix.backwardSupstitution(decomposed, forwardSubstitutedY)

            for i in range(backwardSubstitutedY.getRowsCount()):
                #TODO
                #elementsOfFinalInvertedMatrix[i][j] = backwardSubstitutedY.getElements()[i][0];
                elementsOfFinalInvertedMatrix[i,j] = backwardSubstitutedY.getElements()[i,0]

        finalInvertedMatrix = Matrix(matrixToInvert.getRowsCount(), matrixToInvert.getColsCount(), elementsOfFinalInvertedMatrix)
        return finalInvertedMatrix

    @staticmethod
    def reflect(xc, xh, alpha):
        return Matrix.subtract(Matrix.scalarMultiply(xc, (1 + alpha)), Matrix.scalarMultiply(xh, alpha))
        # return (1 + alpha) * xc - alpha * xh;

class Point:
    def __init__(self, number_of_dimensions, elements):
        self.number_of_dimensions = number_of_dimensions
        self.elements = elements
        #self.matrix = Matrix(number_of_rows = 1, number_of_columns = number_of_dimensions, elements = np.array([elements]))

    def get_value_at_dimension(self, index):
        return self.elements[index]
        #return self.matrix.getElement(row = 0, column = index)

    def set_value_at_dimension(self, index, new_value):
        self.elements[index] = new_value
        #self.matrix = Matrix(number_of_rows = 1, number_of_columns = self.number_of_dimensions, elements = np.array([self.get_elements()]))

    def get_number_of_dimensions(self):
        return self.number_of_dimensions

    def copy(self):
        new_elements = self.get_elements()
        number_of_dimensions = self.get_number_of_dimensions()
        new_point = Point(number_of_dimensions, new_elements)
        return new_point

    def multiply_by_scalar(self, scalar):
        new_point = self.copy()
        number_of_dimensions_of_point = new_point.get_number_of_dimensions()

        for i in range(len(number_of_dimensions_of_point)):
            new_point.set_value_at_dimension(i, new_point.get_value_at_dimension(i) * scalar)

        return new_point



