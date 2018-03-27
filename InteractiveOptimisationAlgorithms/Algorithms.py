import random
from Matrix import *
from Functions import *
from Drawing import *


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


class HookeJeeves(IAlgorithm):

    def __init__(self, f, step, factor, epsilon, print_me):
        self.f = f
        self.step = step
        self.factor = factor
        self.epsilon = epsilon
        self.print_me = print_me

    #@staticmethod
    def run(self, initialPoint):
        #f = open('HookeJeevesOutput.txt', 'w')
        outputString = ""
        xb = Matrix(initialPoint.getRowsCount(), initialPoint.getColsCount(), np.array(initialPoint.getElements()))
        xp = Matrix(initialPoint.getRowsCount(), initialPoint.getColsCount(), np.array(initialPoint.getElements()))
        xn = Matrix(initialPoint.getRowsCount(), initialPoint.getColsCount(), np.array(initialPoint.getElements()))
        iterationNumber = 0
        logger = Logger(self.f)
        while (self.step > self.epsilon):
            additionalInfo = {}
            #nadi xn
            if (self.print_me):
                print "Iteration: " + str(iterationNumber + 1)
            for i in range(xn.getColsCount()):
                plus = Matrix.copyPoint(xn)
                #TODO
                #plus.getElements()[0][i] = plus.getElements()[0][i] + pomak;
                plus.getElements()[0,i] = plus.getElements()[0,i] + self.step
                minus = Matrix.copyPoint(xn)
                #TODO
                #minus.getElements()[0][i] = minus.getElements()[0][i] - pomak;
                minus.getElements()[0,i] = minus.getElements()[0,i] - self.step

                currentMinimum = xn
                test = self.f.valueAt(plus)
                test2 = self.f.valueAt(minus)
                if (self.f.valueAt(plus) < self.f.valueAt(currentMinimum)):
                    currentMinimum = plus
                if (self.f.valueAt(minus) < self.f.valueAt(currentMinimum)):
                    currentMinimum = minus
                xn = currentMinimum

            if (self.print_me):
                print "xb = "
                Matrix.printMatrix(xb)
                print "\txp = "
                Matrix.printMatrix(xp)
                print "\txn = "
                Matrix.printMatrix(xn)
                print "Step = " + str(self.step)

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
                currentIteration = Iteration(iterationNumber, self.f.valueAt(xn), xn.getElement(0, 0), additionalInfo)
            elif(xn.getColsCount() == 2):
                currentIteration = Iteration(iterationNumber, self.f.valueAt(xn), xn, additionalInfo)
            logger.addIteration(currentIteration)

            if (self.f.valueAt(xn) < self.f.valueAt(xb)):
                xp = Matrix.subtract(Matrix.scalarMultiply(xn, 2), xb)

                xb = Matrix.copyPoint(xn)
                xn = Matrix.copyPoint(xp)
            else:
                self.step = self.step * self.factor
                if (self.print_me):
                    print "Changing step to " + str(self.step)
                xp = Matrix.copyPoint(xb)
                xn = Matrix.copyPoint(xp)

            iterationNumber = iterationNumber + 1
            outputString = outputString + str(xn.getElement(0,0))
            for i in range(1, xn.getColsCount()):
                outputString = outputString + " " + str(xn.getElement(0,i))
            outputString = outputString + "\n"
        print "Final solution of Hooke-Jeeves search for initial initialPoint " + str(initialPoint.getElement(0, 0)) + " is " + str(xb.getElements())
        output = open('HookeJeevesOutput.txt', 'w')
        output.write(outputString)
        return xb, logger


class BoxAlgorithm(IAlgorithm):

    def __init__(self, function, lower_bounds, upper_bounds, implicit_constraints, epsilon, alpha, print_me):
        self.function = function
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.implicit_constraints = implicit_constraints
        self.epsilon = epsilon
        self.alpha = alpha
        self.print_me = print_me

    def run(self, point):
        additional_info = {}
        for i in range(len(point.getElements())):
            if(point.getElement(0, i) < self.lower_bounds[i] or point.getElement(0, i) > self.upper_bounds[i]):
                print "Ponudili ste tocku koja ne zadovoljava eksplicitna ogranicenja!"
                return

        n = point.getColsCount()

        centroid = Matrix.copyPoint(point)

        list_of_accepted_points = []

        list_of_accepted_points.append(point)

        for t in range(2*n):
            #elements = new double[1][point.getColsCount()];
            #elements = []
            #elements1 = np.array([[-1.9, 2]])
            elements = np.zeros((1, point.getColsCount()))
            for i in range(point.getColsCount()):
                #double R = ThreadLocalRandom.current().nextDouble(0, 1);
                R = random.uniform(0, 1)
                #elements[0][i] = lower_bounds[i] + R * (upper_bounds[i] - lower_bounds[i])
                #elements.append( [ lower_bounds[i] + R * (upper_bounds[i] - lower_bounds[i]) ] )
                #elements[0][i] = [self.lower_bounds[i] + R * (self.upper_bounds[i] - self.lower_bounds[i])]
                elements[0][i] = self.lower_bounds[i] + R * (self.upper_bounds[i] - self.lower_bounds[i])

            new_point = Matrix(1, point.getColsCount(), elements)
            for j in range(len(self.implicit_constraints)):
                while (not self.implicit_constraints[j].is_satisfied(new_point)):
                    new_point = Matrix.scalarMultiply(Matrix.add(new_point,centroid),0.5)

            list_of_accepted_points.append(new_point)

            #calculate new centroid (with new accepted point)
            #double[][] sum_elements = new double[1][point.getColsCount()];
            #sum_elements = []
            sum_elements = np.zeros((1, point.getColsCount()))
            #for i in range(len(sum_elements[0])):
            #for i in range(len(sum_elements[0])):
                #sum_elements[0][i] = 0
                #sum_elements.append([0])

            sum = Matrix(1, point.getColsCount(), sum_elements)
            for i in range(len(list_of_accepted_points)):
                sum = Matrix.add(sum, list_of_accepted_points[i])
            #centroid = sum/(simplex.length - 2);
            centroid = Matrix.scalarMultiply(sum, (1.0/len(list_of_accepted_points)))

        keepGoing = True
        iteration_number = 1
        logger = Logger(self.function)
        logger.setConstraints(self.implicit_constraints)
        while(keepGoing):
            MIN = float('-inf')
            max = MIN
            valueAtXh = MIN
            valueAtXh2 = MIN
            xhIndex = 0
            xh2Index = 0
            for i in range(len(list_of_accepted_points)):
                #if(function.valueAt(i) > function.valueAt(xhIndex)){
                if(self.function.valueAt(list_of_accepted_points[i]) > self.function.valueAt(list_of_accepted_points[xhIndex])):
                    xh2Index = xhIndex
                    xhIndex = i

            #calculate centroid without xh
            #double[][] sum_elements = new double[1][point.getColsCount()];
            #sum_elements = []
            sum_elements = np.zeros((1, point.getColsCount()))
            sum = Matrix(1, point.getColsCount(), sum_elements)
            #for (int i = 0; i < list_of_accepted_points.size(); i++) {
            for i in range(len(list_of_accepted_points)):
                if( i == xhIndex):
                    pass
                else:
                    sum = Matrix.add(sum, list_of_accepted_points[i])

            centroid = Matrix.scalarMultiply(sum, (1.0/(len(list_of_accepted_points) - 1)))
            xr = Matrix.reflect(centroid,list_of_accepted_points[xhIndex], self.alpha)
            for i in range(n):
                if(xr.getElement(0,i) < self.lower_bounds[i]):
                    xr.getElements()[0][i] = self.lower_bounds[i]
                elif(xr.getElements()[0][i] > self.upper_bounds[i]):
                    xr.getElements()[0][i] = self.upper_bounds[i]

            for i in range(len(self.implicit_constraints)):
                while (not self.implicit_constraints[i].is_satisfied(xr)):
                    xr = Matrix.scalarMultiply(Matrix.add(xr,centroid),0.5)

            if(self.function.valueAt(xr) > self.function.valueAt(list_of_accepted_points[xh2Index])):
                xr = Matrix.scalarMultiply(Matrix.add(xr,centroid),0.5)

            #Matrix[] arrayPrihvacenihTocaka = list_of_accepted_points.toArray(new Matrix[]{});
            arrayPrihvacenihTocaka = list_of_accepted_points
            arrayPrihvacenihTocaka[xhIndex] = xr
            #list_of_accepted_points = new LinkedList<Matrix>();
            list_of_accepted_points = []
            for i in range(len(arrayPrihvacenihTocaka)):
                list_of_accepted_points.append(arrayPrihvacenihTocaka[i])

            keepGoing = False
            for i in range(len(list_of_accepted_points)):
                if(abs(self.function.valueAt(list_of_accepted_points[i]) - self.function.valueAt(centroid)) > self.epsilon):
                    keepGoing = True


            #TODO check if this is the correct place to log the additional_info points

            xhDescription = "xh - The point in which the function value is highest"
            xrDescription = "xr - Reflected point"
            xcDescription = "xc - Centroid"

            xhTuple = (list_of_accepted_points[xhIndex], xhDescription)
            xrTuple = (xr, xrDescription)
            xcTuple = (centroid, xcDescription)

            additional_info["xh"] = xhTuple
            additional_info["xr"] = xrTuple
            additional_info["xc"] = xcTuple

            # currentIteration = Iteration(iteracija, f.valueAt(xn), xn, additional_info)
            # currentIteration = Iteration(iteracija, f.valueAt(xn), xn.getElement(0, 0), additional_info)
            if (centroid.getColsCount() == 1):
                currentIteration = Iteration(iteration_number, self.function.valueAt(centroid), centroid.getElement(0, 0), additional_info)
            elif (centroid.getColsCount() == 2):
                currentIteration = Iteration(iteration_number, self.function.valueAt(centroid), centroid, additional_info)
            logger.addIteration(currentIteration)

            iteration_number = iteration_number + 1

        return centroid, logger