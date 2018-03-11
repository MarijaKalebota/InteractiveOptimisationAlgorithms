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
import numbers
import decimal
#from Algorithms import *
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
        #self.xValue = xValue
        #if(isinstance(xValue, float)):
        if (isinstance(xValue, numbers.Number)):
            self.xValue = xValue
            self.x1Value = xValue
            self.x2Value = xValue
        else:
            self.x1Value = xValue.getElements()[0,0]
            self.x2Value = xValue.getElements()[0,1]
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

    #def drawAnimation(self):
    def drawAnimation(self, minX, maxX, minY, maxY, numberOfSamplesOfX):
        playMaxOfInterval = 0
        arrayX = []
        arrayY = []
        '''
        minX = self.logger.getIterations()[0].xValue
        maxX = self.logger.getIterations()[0].xValue
        minY = self.logger.getIterations()[0].yValue
        maxY = self.logger.getIterations()[0].yValue
        '''

        for iteration in self.logger.getIterations():
            playMaxOfInterval = playMaxOfInterval + 1
            # stvori polje ovih brojeva
            arrayX.append(iteration.xValue)
            '''
            if(iteration.xValue < minX):
                minX = iteration.xValue
            if (iteration.xValue > maxX):
                maxX = iteration.xValue
            if (iteration.yValue < minY):
                minY = iteration.yValue
            if (iteration.yValue > maxY):
                maxY = iteration.yValue
            '''

            arrayY.append(iteration.yValue)
        '''
        print "ArrayX: "
        for x in arrayX:
            print str(x)
        print "ArrayY: "
        for x in arrayY:
            print str(x)
        '''

        relativeDistanceX = (maxX - minX) / 10
        relativeDistaceY = (maxY - minY) / 10

        w = widgets.IntSlider(min=0, max=playMaxOfInterval - 1, step=1, value=0)
        play = widgets.Play(
            value=0,
            min=0,
            max=playMaxOfInterval,
            step=1,
            description="Press play",
            disabled=False
        )
        nextButton = widgets.Button(description="Next")
        previousButton = widgets.Button(description="Previous")

        def on_nextButton_clicked(x):
            if(play.value < play.max):
                play.value += 1

        def on_previousButton_clicked(x):
            if(play.value > 0):
                play.value -= 1

        nextButton.on_click(on_nextButton_clicked)
        previousButton.on_click(on_previousButton_clicked)

        widgets.jslink((play, 'value'), (w, 'value'))
        #def f(arrayX, arrayY, iteration):
        #def f(arrayX, arrayY, iteration, minX, maxX, minY, maxY, relOdmakX, relOdmakY):
        def f(arrayX, arrayY, iteration, minX, maxX, minY, maxY):
            function = self.logger.getFunction()
            plt.clf()
            plt.close('all')
            plt.figure(iteration)
            #plt.axis([0.0, 15.0, -5.0, 30.0])
            #plt.axis([0.0, 6.0, -0.5, 4.0])
            #TODO remove hardcoded axis - set it to (min(x) - relative distance),(max(x) + relative distance), same for y
            #plt.axis([minX - relDistanceX, maxX + relDistanceX, minY - relDistanceY, maxY + relDistanceY])
            plt.axis([minX, maxX, minY, maxY])
            ax = plt.gca()
            ax.set_autoscale_on(False)

            #X = np.linspace(0.0, 13.0, num=10)

            X = np.linspace(minX, maxX, num=numberOfSamplesOfX)
            #X = np.linspace(minX - relDistanceX, maxX + relDistanceX, num=10)
            #X = np.linspace(0.0, 6.0, num=10)
            #TODO set this linspace to the same values as axis
            Y = [function.valueAt(x) for x in X]

            plt.plot(X, Y, 'b')
            plt.plot(arrayX[iteration], arrayY[iteration], 'ro')
            plt.show()

        #interact(f, arrayX=fixed(arrayX), arrayY = fixed(arrayY), iteration=w)
        #interact(f, arrayX=fixed(arrayX), arrayY=fixed(arrayY), iteration=w, minX=fixed(minX), maxX=fixed(maxX), minY=fixed(minY), maxY=fixed(maxY), relDistanceX = fixed(relativeDistanceX), relDistanceY = fixed(relativeDistaceY))
        interact(f, arrayX=fixed(arrayX), arrayY=fixed(arrayY), iteration=w, minX=minX, maxX=maxX, minY=minY, maxY=maxY)
        display(play)

        display(previousButton)
        display(nextButton)

    #def drawAnimationContour(self):
    def drawAnimationContour(self, minX, maxX, minY, maxY, numberOfSamplesOfX):
        playMaxOfInterval = 0
        poljeX = []
        poljeY = []

        for iteration in self.logger.getIterations():
            playMaxOfInterval = playMaxOfInterval + 1
            # stvori polje ovih brojeva
            #poljeX.append(iteration.xValue)
            #poljeX.append(iteration.xValue)
            poljeX.append(iteration.x1Value)
            #poljeX.append(iteration.xValue.getElements()[0,0])
            #poljeY.append(iteration.yValue)
            poljeY.append(iteration.x2Value)
            #poljeY.append(iteration.yValue)

        '''
        print "PoljeX: "
        for x in poljeX:
            print str(x)
        print "PoljeY: "
        for x in poljeY:
            print str(x)
        '''

        w = widgets.IntSlider(min=0, max=playMaxOfInterval - 1, step=1, value=0)
        play = widgets.Play(
            value=0,
            min=0,
            max=playMaxOfInterval,
            step=1,
            description="Press play",
            disabled=False
        )
        nextButton = widgets.Button(description="Next")
        previousButton = widgets.Button(description="Previous")

        def on_nextButton_clicked(x):
            if (play.value < play.max):
                play.value += 1

        def on_previousButton_clicked(x):
            if (play.value > 0):
                play.value -= 1

        nextButton.on_click(on_nextButton_clicked)
        previousButton.on_click(on_previousButton_clicked)

        widgets.jslink((play, 'value'), (w, 'value'))

        def f(poljeX, poljeY, poljeZ, xRjesenja, yRjesenja, iteration):
        #def f(poljeX, poljeY, poljeZ, xRjesenja, yRjesenja, iteration, minX, maxX, minY, maxY):
            # def f(poljeX, poljeY, iteration, minX, maxX, minY, maxY, relOdmakX, relOdmakY):
            funkcija = self.logger.getFunction()
            plt.clf()
            plt.close('all')
            plt.figure(iteration)
            #plt.axis([-6.0, 6.0, -6.0, 6.0])
            # plt.axis([0.0, 6.0, -0.5, 4.0])
            # TODO maknuti hardkodirani axis - postaviti ga na (min(x) - relativni odmak),(max(x) + relativni odmak), isto za y
            # plt.axis([minX - relOdmakX, maxX + relOdmakX, minY - relOdmakY, maxY + relOdmakY])
            plt.axis([minX, maxX, minY, maxY])
            ax = plt.gca()
            ax.set_autoscale_on(False)

            #X = np.linspace(-6.0, 6.0, num=10)
            X = np.linspace(minX, maxX, num=numberOfSamplesOfX)
            # X = np.linspace(minX - relOdmakX, maxX + relOdmakX, num=10)
            # X = np.linspace(0.0, 6.0, num=10)
            # TODO ovaj linspace staviti na iste vrijednosti kao i axis
            Y = [funkcija.valueAt(x) for x in X]

            plt.contourf(poljeX, poljeY, poljeZ, 20, cmap='RdGy')
            plt.colorbar();

            #plt.plot(X, Y, 'b')
            #plt.plot(poljeX[iteration], poljeY[iteration], 'bo')
            plt.plot(xRjesenja[iteration], yRjesenja[iteration], 'go')
            plt.show()


        def rosenbrock(x, y):
            # return np.sin(np.sqrt(x ** 2 + y ** 2))
            return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

        #TODO get file name as argument, or from logger!
        thisFile = open('HookeJeevesOutput.txt', 'r')
        lines = thisFile.readlines()
        # print lines
        for i in range(len(lines)):
            lines[i] = lines[i].split()
            # lines[i] = lines[i][0].split(" ")
        # print lines
        thisFile.close()
        rjesenja = lines

        xRjesenja = []
        yRjesenja = []

        for rjesenje in rjesenja:
            xRjesenja.append(float(rjesenje[0]))
            yRjesenja.append(float(rjesenje[1]))

        #x = np.linspace(-6, 6, 30)
        x = np.linspace(minX, maxX, 30)
        #y = np.linspace(-6, 6, 30)
        y = np.linspace(minY, maxY, 30)

        X, Y = np.meshgrid(x, y)
        Z = rosenbrock(X, Y)

        #interact(f, poljeX=fixed(X), poljeY=fixed(Y), poljeZ = fixed(Z), iteration=w)
        # interact(f, poljeX=fixed(poljeX), poljeY=fixed(poljeY), iteration=w, minX=fixed(minX), maxX=fixed(maxX), minY=fixed(minY), maxY=fixed(maxY), relOdmakX = fixed(relativniOdmakX), relOdmakY = fixed(relativniOdmaky))
        interact(f, poljeX=fixed(X), poljeY=fixed(Y), poljeZ=fixed(Z), xRjesenja = fixed(xRjesenja), yRjesenja = fixed(yRjesenja), iteration=w)
        #interact(f, poljeX=fixed(X), poljeY=fixed(Y), poljeZ=fixed(Z), xRjesenja=fixed(xRjesenja), yRjesenja=fixed(yRjesenja), iteration=w, minX = minX, maxX = maxX, minY = minY, maxY = maxY)
        display(play)

        display(previousButton)
        display(nextButton)


# #main
# def main():
#     f3OneDimensional1 = F3OneDimensional()
#     f3OneDimensional12 = F3OneDimensional()
#     elements = np.array([[10]])
#     point1 = Matrix(1, 1, elements)
#     point12 = Matrix.copyPoint(point1)
#     interval = GoldenSection.runGoldenSectionWithInitialPoint(point1.getElement(0, 0), 1E-6, f3OneDimensional1, True)
#     h_j_algorithm = HookeJeeves(f3OneDimensional12, 1, 0.5, 1E-6, True)
#     #solutionHookeJeeves, loggerHookeJeeves = HookeJeeves.provediHookeJeeves(f3OneDimensional12, point12, 1, 0.99, 1E-6, True)
#     #solutionHookeJeeves, loggerHookeJeeves = HookeJeeves.run(f3OneDimensional12, point12, 1, 0.5, 1E-6, True)
#     solutionHookeJeeves, loggerHookeJeeves = h_j_algorithm.run(point12)
#
#     #print point1.getRowsCount()
#     #Matrix.printMatrix(solutionHookeJeeves)
#
# if (__name__ == '__main__'):
#     main()
