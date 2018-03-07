from Functions import *
from Test0 import *


class IAlgorithm(object):

    def run(self, initialPoint):
        raise NotImplementedError


class Unimodal:
    @staticmethod
    def findUnimodal(point, h, f): #point, h, f):
        #valueAtLeftOfPoint = f.valueAt(point - 2 * h);
        #valueAtLeftOfPoint = f(point - 2 * h)
        valueAtLeftOfPoint = f.valueAt(point - 2 * h)
        valueAtPoint = f.valueAt(point)
        valueAtRightOfPoint = f.valueAt(point + 2 * h)
        power = 1
        #returnValue = new double[2];
        returnValue = [0, 0]
        if (valueAtLeftOfPoint < valueAtPoint):
        # use the formula with the minus
            rightSideOfInterval = point
            middleOfInterval = point - 2 * h
            leftSideOfInterval = point - 2 * 2 * h
            #while (valueAtPoint > valueAtLeftOfPoint){
            while (f.valueAt(middleOfInterval) > f.valueAt(leftSideOfInterval)):
                #rightSideOfInterval = point - Math.pow(2, power) * h;
                rightSideOfInterval = point - (2 ** power) * h
                #middleOfInterval = point - Math.pow(2, power+1) * h;
                middleOfInterval = point - (2 ** (power + 1)) * h
                #leftSideOfInterval = point - Math.pow(2, power+2) * h;
                leftSideOfInterval = point - (2 ** (power + 2)) * h

                power = power + 1


            returnValue[0] = leftSideOfInterval
            returnValue[1] = rightSideOfInterval
            return returnValue


        elif (valueAtRightOfPoint < valueAtPoint):
            leftSideOfInterval = point
            middleOfInterval = point + 2 * h
            rightSideOfInterval = point + 2 * 2 * h

            while (f.valueAt(middleOfInterval) > f.valueAt(rightSideOfInterval)):
                #leftSideOfInterval = point + Math.pow(2, power) * h;
                leftSideOfInterval = point + (2 ** power) * h
                #middleOfInterval = point + Math.pow(2, power+1) * h;
                middleOfInterval = point + (2 ** (power + 1)) * h
                #rightSideOfInterval = point + Math.pow(2, power+2) * h;
                rightSideOfInterval = point + (2 ** (power + 2)) * h
                power = power + 1


            returnValue[0] = leftSideOfInterval
            returnValue[1] = rightSideOfInterval
            return returnValue


        else:
            returnValue[0] = point - 2 * h
            returnValue[1] = point + 2 * h
            return returnValue


class GoldenSection:
    #EPSILON = 1E-6
    @staticmethod
    def runGoldenSectionWithInitialPoint(point, epsilon, f, printMe): #IFunction f)
        if (printMe):
            print "Running golden section"
        unimodalInterval = Unimodal.findUnimodal(point, 1, f)
        if (printMe):
            print "Found unimodal interval"

        a = unimodalInterval[0]
        b = unimodalInterval[1]
        if (printMe):
            print "Unimodal interval. a = " + str(a) + "\tb = " + str(b)


        #double k = (Math.sqrt(5) - 1) / 2.; // k = 0.618
        k = (5**(0.5) - 1) / 2. # k = 0.618
        if (printMe):
            print "k = " + str(k)


        c = b - (b-a) * k
        d = a + (b-a) * k
        if (printMe):
            print "c = " + str(c) + "\td = " + str(d)
        iteracija = 1
        while (abs(a - b) > epsilon):
            if (printMe):
                print "Iteracija: " + str(iteracija)
                print "c = " + str(c) + "\td = " + str(d)

            if (f.valueAt(c) <= f.valueAt(d)):
                #move c to d, d to b, calculate new c
                if (printMe):
                    print "c is better than d"
                b = d
                d = c
                c = b - (b-a) * k

            elif (f.valueAt(c) > f.valueAt(d)):
                #move d to c, c to a, calculate new d
                if (printMe):
                    print "d is better than c"
                a = c
                c = d
                d = a + (b-a) * k

            else:
                print "Neither of the two conditions was met in the golden section search"

            if (printMe):
                print "New a = " + str(a) + "\tc = " + str(c) + "\td = " + str(d) + "\t b = " + str(b)

            iteracija = iteracija + 1


        returnValue = [0, 0]
        returnValue[0] = a
        returnValue[1] = b
        print "\nGolden section - final solution: [" + str(a) + "]["+ str(b) + "]"
        return returnValue

    @staticmethod
    def runGoldenSectionWithKnownInterval(a, b, f, epsilon):#, IFunction f){
        #k = (1. + Math.sqrt(5)) / 2.; // k = 0.618
        k = (5 ** (0.5) - 1) / 2.  # k = 0.618
        c = b - (b - a) * k
        d = a + (b - a) * k
        while (abs(a - b) > epsilon):
            if (f.valueAt(c) <= f.valueAt(d)):
                # move c to d, d to b, calculate new c
                b = d
                d = c
                c = b - (b-a) * k

            elif (f.valueAt(c) > f.valueAt(d)):
                #move d to c, c to a, calculate new d
                a = c
                c = d
                d = a + (b-a) * k


        returnValue = [0,0]
        returnValue[0] = a
        returnValue[1] = b
        return returnValue


class HookeJeeves:
    @staticmethod
    def run(f, point, step, factor, epsilon, printMe):
        #f = open('HookeJeevesOutput.txt', 'w')
        outputString = ""
        xb = Matrix(point.getRowsCount(), point.getColsCount(), np.array(point.getElements()))
        xp = Matrix(point.getRowsCount(), point.getColsCount(), np.array(point.getElements()))
        xn = Matrix(point.getRowsCount(), point.getColsCount(), np.array(point.getElements()))
        iterationNumber = 0
        logger = Logger(f)
        while (step > epsilon):
            additionalInfo = {}
            #nadi xn
            if (printMe):
                print "Iteration: " + str(iterationNumber + 1)
            for i in range(xn.getColsCount()):
                plus = Matrix.copyPoint(xn)
                #TODO
                #plus.getElements()[0][i] = plus.getElements()[0][i] + pomak;
                plus.getElements()[0,i] = plus.getElements()[0,i] + step
                minus = Matrix.copyPoint(xn)
                #TODO
                #minus.getElements()[0][i] = minus.getElements()[0][i] - pomak;
                minus.getElements()[0,i] = minus.getElements()[0,i] - step

                currentMinimum = xn
                test = f.valueAt(plus)
                test2 = f.valueAt(minus)
                if (f.valueAt(plus) < f.valueAt(currentMinimum)):
                    currentMinimum = plus
                if (f.valueAt(minus) < f.valueAt(currentMinimum)):
                    currentMinimum = minus
                xn = currentMinimum

            if (printMe):
                print "xb = "
                Matrix.printMatrix(xb)
                print "\txp = "
                Matrix.printMatrix(xp)
                print "\txn = "
                Matrix.printMatrix(xn)
                print "Step = " + str(step)

            xbDescription = "xb - Bazna tocka algoritma Hooke-Jeeves"
            xpDescription = "xp - Tocka pretrazivanja algoritma Hooke-Jeeves u trenutnoj iteraciji. xp' = 2*xn + xb"
            xnDescription = "xn - Nova tocka algoritma Hooke-Jeeves, izracunata (dobivena pretrazivanjem) u trenutnoj iteraciji, Ona u sljedecoj iteraciji postaje bazna tocka."

            xbTuple = (xb, xbDescription)
            xpTuple = (xp, xpDescription)
            xnTuple = (xn, xnDescription)

            additionalInfo["xb"] = xbTuple
            additionalInfo["xp"] = xpTuple
            additionalInfo["xn"] = xnTuple

            #currentIteration = Iteration(iteracija, f.valueAt(xn), xn, additionalInfo)
            #currentIteration = Iteration(iteracija, f.valueAt(xn), xn.getElement(0, 0), additionalInfo)
            if(xn.getColsCount() == 1):
                currentIteration = Iteration(iterationNumber, f.valueAt(xn), xn.getElement(0, 0), additionalInfo)
            elif(xn.getColsCount() == 2):
                currentIteration = Iteration(iterationNumber, f.valueAt(xn), xn, additionalInfo)
            logger.addIteration(currentIteration)

            if (f.valueAt(xn) < f.valueAt(xb)):
                xp = Matrix.subtract(Matrix.scalarMultiply(xn, 2), xb)

                xb = Matrix.copyPoint(xn)
                xn = Matrix.copyPoint(xp)
            else:
                step = step * factor
                if (printMe):
                    print "Changing step to " + str(step)
                xp = Matrix.copyPoint(xb)
                xn = Matrix.copyPoint(xp)

            iterationNumber = iterationNumber + 1
            outputString = outputString + str(xn.getElement(0,0))
            for i in range(1, xn.getColsCount()):
                outputString = outputString + " " + str(xn.getElement(0,i))
            outputString = outputString + "\n"
        print "Final solution of Hooke-Jeeves search for initial point " + str(point.getElement(0, 0)) + " is " + str(xb.getElements())
        output = open('HookeJeevesOutput.txt', 'w')
        output.write(outputString)
        return xb, logger