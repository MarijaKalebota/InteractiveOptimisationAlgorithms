#import Matrix
from Matrix import *
#import Functions
from Functions import *
import sys
import numpy as np
import scipy as sp
import sklearn
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
#%pylab inline
#from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.utils.py3compat import annotate
from IPython.display import display
#from ipywidgets import FloatSlider
from ipywidgets import *
#class IFunction:

'''
def f(x):
    #return Math.pow(point.getElement(0, 0) - 4, 2) + 4 * Math.pow(point.getElement(0, 1) - 2, 2);
    return (x - 3)**2
'''

class Iteration():
    def __init__(self, iterationNumber, yValue, xValue, additionalInfo):
        self.iterationNumber = iterationNumber
        self.yValue = yValue
        self.xValue = xValue
        self.additionalInfo = additionalInfo

class Logger(object):
    def __init__(self, f):
        self.f = f
        self.iterations = []

    def addIteration(self, iteration):
        self.iterations.append(iteration)

    def getIterations(self):
        return self.iterations

    def getIteration(self, indexOfIterationToGet):
        if(len(self.iterations) >= indexOfIterationToGet):
            return "Index out of range"
        else:
            return self.iterations[indexOfIterationToGet]

    def getFunction(self):
        return self.f;

    def printLogToFile(self, file):
        #TODO
        return

    def printLogToOutput(self):
        #TODO
        '''
        for iteration in self.iterations:
            for element in iteration.additionalInfo:
                if type(element[1] )
        outputString = outputString + str(xn.getElement(0, 0))
        for i in range(1, xn.getColsCount()):
            outputString = outputString + " " + str(xn.getElement(0, i))
        outputString = outputString + "\n"

        print "Konacno rjesenje Hooke-Jeevesa za pocetnu tocku " + str(tocka.getElement(0, 0)) + " je " + str(
            xb.getElements())
        output = open('HookeJeevesOutput.txt', 'w')
        output.write(outputString)
        '''
        return

class Drawer:
    def __init__(self, logger):
        self.logger = logger

    def drawAnimation(self):
        playMaxOfInterval = 0
        poljeX = []
        poljeY = []
        for iteration in self.logger.getIterations():
            playMaxOfInterval = playMaxOfInterval + 1
            # stvori polje ovih brojeva
            poljeX.append(iteration.xValue)
            poljeY.append(iteration.yValue)

        w = widgets.IntSlider(min=0, max=playMaxOfInterval - 1, step=1, value=0)
        play = widgets.Play(
            value=0,
            min=0,
            max=playMaxOfInterval,
            step=1,
            description="Press play",
            disabled=False
        )
        widgets.jslink((play, 'value'), (w, 'value'))
        def f(poljeX, poljeY, index):
            funkcija = self.logger.getFunction()
            plt.clf()
            plt.close('all')
            plt.figure(index)
            plt.axis([0.0, 15.0, -5.0, 30.0])
            #TODO maknuti hardkodirani axis - postaviti ga na (min(x) - relativni odmak),(max(x) + relativni odmak), isto za y
            ax = plt.gca()
            ax.set_autoscale_on(False)

            X = np.linspace(0.0, 13.0, num=10)
            #TODO ovaj linspace staviti na iste vrijednosti kao i axis
            Y = [funkcija.valueAt(x) for x in X]

            plt.plot(X, Y, 'b')
            plt.plot(poljeX[index], poljeY[index], 'ro')
            plt.show()

        interact(f, poljeX=fixed(poljeX), poljeY = fixed(poljeY), index=w)
        display(play)


class Unimodal:
    @staticmethod
    def findUnimodal(tocka, h, f): #tocka, h, f):
        #valueAtLeftOfTocka = f.valueAt(tocka - 2 * h);
        #valueAtLeftOfTocka = f(tocka - 2 * h)
        valueAtLeftOfTocka = f.valueAt(tocka - 2 * h)
        valueAtTocka = f.valueAt(tocka)
        valueAtRightOfTocka = f.valueAt(tocka + 2 * h)
        power = 1
        #returnValue = new double[2];
        returnValue = [0, 0]
        if (valueAtLeftOfTocka < valueAtTocka):
        # koristi formulu s minusom
            rightSideOfInterval = tocka
            middleOfInterval = tocka - 2 * h
            leftSideOfInterval = tocka - 2 * 2 * h
            #while (valueAtTocka > valueAtLeftOfTocka){
            while (f.valueAt(middleOfInterval) > f.valueAt(leftSideOfInterval)):
                #rightSideOfInterval = tocka - Math.pow(2, power) * h;
                rightSideOfInterval = tocka - (2**power) * h
                #middleOfInterval = tocka - Math.pow(2, power+1) * h;
                middleOfInterval = tocka - (2**(power + 1)) * h
                #leftSideOfInterval = tocka - Math.pow(2, power+2) * h;
                leftSideOfInterval = tocka - (2**(power + 2)) * h

                power = power + 1


            returnValue[0] = leftSideOfInterval
            returnValue[1] = rightSideOfInterval
            return returnValue


        elif (valueAtRightOfTocka < valueAtTocka):
            leftSideOfInterval = tocka
            middleOfInterval = tocka + 2 * h
            rightSideOfInterval = tocka + 2 * 2 * h

            while (f.valueAt(middleOfInterval) > f.valueAt(rightSideOfInterval)):
                #leftSideOfInterval = tocka + Math.pow(2, power) * h;
                leftSideOfInterval = tocka +(2**power) * h
                #middleOfInterval = tocka + Math.pow(2, power+1) * h;
                middleOfInterval = tocka +(2**(power + 1)) * h
                #rightSideOfInterval = tocka + Math.pow(2, power+2) * h;
                rightSideOfInterval = tocka + (2**(power + 2)) * h
                power = power + 1


            returnValue[0] = leftSideOfInterval
            returnValue[1] = rightSideOfInterval
            return returnValue


        else:
            returnValue[0] = tocka - 2 * h
            returnValue[1] = tocka + 2 * h
            return returnValue


class ZlatniRez:
    #EPSILON = 1E-6
    @staticmethod
    def provediZlatniRezSPocTockom(tocka, epsilon, f, printMe): #IFunction f)
        if (printMe):
            print "Usao sam u provodenje zlatnog reza"
        unimodalniInterval = Unimodal.findUnimodal(tocka, 1, f)
        if (printMe):
            print "Izasao sam iz trazenja unimodalnog intervala"

        a = unimodalniInterval[0]
        b = unimodalniInterval[1]
        if (printMe):
            print "Unimodalni interval. a = " + str(a) + "\tb = " + str(b)


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
                #pomakni c na d, d na b, izracunaj novi c
                if (printMe):
                    print "c je bolji od d"
                b = d
                d = c
                c = b - (b-a) * k

            elif (f.valueAt(c) > f.valueAt(d)):
                #pomakni d na c, c na a, izracunaj novi d
                if (printMe):
                    print "d je bolji od c"
                a = c
                c = d
                d = a + (b-a) * k

            else:
                print "U zlatnom rezu nije zadovoljen nijedan od dva uvjeta"

            if (printMe):
                print "Novi a = " + str(a) + "\tc = " + str(c) + "\td = " + str(d) + "\t b = " + str(b)

            iteracija = iteracija + 1


        returnValue = [0, 0]
        returnValue[0] = a
        returnValue[1] = b
        print "\nKonacni rezultat zlatnog reza: [" + str(a) + "]["+ str(b) + "]"
        return returnValue

    @staticmethod
    def provediZlatniRezSPoznatimIntervalom(a, b, f, epsilon):#, IFunction f){
        #k = (1. + Math.sqrt(5)) / 2.; // k = 0.618
        k = (5 ** (0.5) - 1) / 2.  # k = 0.618
        c = b - (b - a) * k
        d = a + (b - a) * k
        while (abs(a - b) > epsilon):
            if (f.valueAt(c) <= f.valueAt(d)):
                # pomakni c na d, d na b, izracunaj novi c
                b = d
                d = c
                c = b - (b-a) * k

            elif (f.valueAt(c) > f.valueAt(d)):
                #pomakni d na c, c na a, izracunaj novi d
                a = c
                c = d
                d = a + (b-a) * k


        returnValue = [0,0]
        returnValue[0] = a
        returnValue[1] = b
        return returnValue




class HookeJeeves:
    @staticmethod
    def provediHookeJeeves(f, tocka, pomak, factor, epsilon, printMe):
        #f = open('HookeJeevesOutput.txt', 'w')
        outputString = ""
        xb = Matrix(tocka.getRowsCount(), tocka.getColsCount(), np.array(tocka.getElements()))
        xp = Matrix(tocka.getRowsCount(), tocka.getColsCount(), np.array(tocka.getElements()))
        xn = Matrix(tocka.getRowsCount(), tocka.getColsCount(), np.array(tocka.getElements()))
        iteracija = 0
        logger = Logger(f)
        while (pomak > epsilon):
            additionalInfo = {}
            #nadi xn
            if (printMe):
                print "Iteracija: " + str(iteracija + 1)
            for i in range(xn.getColsCount()):
                plus = Matrix.copyPoint(xn)
                #TODO
                #plus.getElements()[0][i] = plus.getElements()[0][i] + pomak;
                plus.getElements()[0,i] = plus.getElements()[0,i] + pomak
                minus = Matrix.copyPoint(xn)
                #TODO
                #minus.getElements()[0][i] = minus.getElements()[0][i] - pomak;
                minus.getElements()[0,i] = minus.getElements()[0,i] - pomak

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
                print "Pomak = " + str(pomak)

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
            currentIteration = Iteration(iteracija, f.valueAt(xn), xn.getElement(0, 0), additionalInfo)
            logger.addIteration(currentIteration)

            if (f.valueAt(xn) < f.valueAt(xb)):
                xp = Matrix.subtract(Matrix.scalarMultiply(xn, 2), xb)

                xb = Matrix.copyPoint(xn)
                xn = Matrix.copyPoint(xp)
            else:
                pomak = pomak * factor
                if (printMe):
                    print "Changing pomak to " + str(pomak)
                xp = Matrix.copyPoint(xb)
                xn = Matrix.copyPoint(xp)

            iteracija = iteracija + 1
            outputString = outputString + str(xn.getElement(0,0))
            for i in range(1, xn.getColsCount()):
                outputString = outputString + " " + str(xn.getElement(0,i))
            outputString = outputString + "\n"
        print "Konacno rjesenje Hooke-Jeevesa za pocetnu tocku " + str(tocka.getElement(0,0)) + " je " + str(xb.getElements())
        output = open('HookeJeevesOutput.txt', 'w')
        output.write(outputString)
        return xb, logger

#main

def main():
    f3OneDimensional1 = F3OneDimensional()
    f3OneDimensional12 = F3OneDimensional()
    elements = np.array([[10]])
    tocka1 = Matrix(1, 1, elements)
    tocka12 = Matrix.copyPoint(tocka1)
    interval = ZlatniRez.provediZlatniRezSPocTockom(tocka1.getElement(0,0), 1E-6, f3OneDimensional1, True)
    #rjesenjeHookeJeeves, loggerHookeJeeves = HookeJeeves.provediHookeJeeves(f3OneDimensional12, tocka12, 1, 0.99, 1E-6, True)
    rjesenjeHookeJeeves, loggerHookeJeeves = HookeJeeves.provediHookeJeeves(f3OneDimensional12, tocka12, 1, 0.5, 1E-6, True)

    #print tocka1.getRowsCount()
    #Matrix.printMatrix(rjesenjeHookeJeeves)

if (__name__ == '__main__'):
    main()
