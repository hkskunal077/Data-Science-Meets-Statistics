def lecture4():
    import random
    import math

    def sameDate(numPeople, numSame):
        possibleDates = range(366)
        birthdays = [0]*366
        for person in range(numPeople):
            birthDate = random.choice(possibleDates)
            birthdays[birthDate] += 1
        return max(birthdays) >= numSame

    def estProb_sameDate(numPeople, numSame, numTrials = 100):
        count = 0
        for it in range(numTrials):
            if sameDate(numPeople, numSame) == True:
                count += 1

        return count/numTrials

     ##print(estProb_sameDate(23, 2))
    ##print(estProb_sameDate(23, 2, 1000))
    ##print(estProb_sameDate(23, 2, 10000))


    def wrapperMultipleTests():
        for numPeople in [10, 20, 40, 100]:
            print('For', numPeople,'est. prob. of a shared birthday is',
              round((estProb_sameDate(numPeople, 2, 100000)), 3))
            numerator = math.factorial(366)
            denom = (366**numPeople)*math.factorial(366-numPeople)
            print('Actual prob. for N = 100 =',
              round((1 - numerator/denom), 3), "\n")

    #It is easy to scale this Mathematical property by programming as compared to
    #actual math because computer can do brute force much faster.
        
        
        
    #We cannot adjust the Mathematical formula, but simulation can be adjusted
    #So also we cannot create True Biasedness, so we add relative weights

    def sameDate(numPeople, numSame):
        possibleDates = 4*list(range(0, 57)) + [58] + 4*list(range(59, 366))\
                        + 4*list(range(180, 270))
        
        birthdays = [0]*366
        for person in range(numPeople):
            birthDate = random.choice(possibleDates)
            birthdays[birthDate] += 1
        return max(birthdays) >= numSame

    wrapperMultipleTests()
#lecture4()


def lecture5():

    # we will need class of drunk person, class of location, class of field
    import matplotlib.pyplot as plt
    import pylab
    import random

    #These are custom styling for matplotlib 
    #set line width
    plt.rcParams['lines.linewidth'] = 4
    #set font size for titles 
    plt.rcParams['axes.titlesize'] = 20
    #set font size for labels on axes
    plt.rcParams['axes.labelsize'] = 20
    #set size of numbers on x-axis
    plt.rcParams['xtick.labelsize'] = 16
    #set size of numbers on y-axis
    plt.rcParams['ytick.labelsize'] = 16




    class Location(object):
        def __init__ (self, x, y):
            self.x = x
            self.y = y

        def move(self, deltaX, deltaY):
            return Location(self.x + deltaX, self.y + deltaY)

        def getX(self):
            return self.x

        def getY(self):
            return self.y

        def distFrom(self, other):
            xDist = self.x - other.getX() #accessing through public member functions
            yDist = self.y - other.getY()
            return (xDist**2 + yDist**2)**0.5

        def __str__ (self):
            return '<' + str(self.x) + ', ' + str(self.y)+ '>'



    class Field(object):
        def __init__ (self):
            self.drunks = {}

        def addDrunk(self, drunk, loc):
            if drunk in self.drunks:
                raise ValueError('Duplicate Drunk')
            else:
                self.drunks[drunk] = loc

        def moveDrunk(self, drunk):
            if drunk not in self.drunks:
                raise ValueError('Drunk is not in the field')

            xDist, yDist = drunk.takeStep()
            self.drunks[drunk] =\
                               self.drunks[drunk].move(xDist, yDist)

        def getLoc(self, drunk):
            if drunk not in self.drunks:
                raise ValueError('Drunk not in Field')
            return self.drunks[drunk]




    class Drunk(object):
        def __init__ (self, name = None):
            self.name = name

        def __str__ (self):
            if self != None:
                return self.name
            return 'Anonymous'


    class UsualDrunk(Drunk):
        def takeStep(self):
            stepChoices = [(0,1),(0, -1),(1, 0),(-1, 0)]
            return random.choice(stepChoices)


    class MasochisrDrunk(Drunk):
        def takeStep(self):
            stepChoices = [(0.0, 1.1), (0.0, -0.9), (1.0, 0.0), (-1.0, 0.0)]
            return random.choice(stepChoices)


    #A singular walk
    def walk(f, d, numSteps):
        """Assumes: f a Field, d a Drunk in f, and numSteps an int >= 0.
           Moves d numSteps times, and returns the distance between
           the final location and the location at the start of the 
           walk."""
        start = f.getLoc(d)
        for s in range(numSteps):
            f.moveDrunk(d)
        return start.distFrom(f.getLoc(d))

    #Simulating the same walk again and again everytime init at origin(0, 0)
    def simWalks(numSteps, numTrials, dClass):
        """Assumes numSteps an int >= 0, numTrials an int > 0,
             dClass a subclass of Drunk
           Simulates numTrials walks of numSteps steps each.
           Returns a list of the final distanceñs for each trial"""
        Homer = dClass('Homer')
        origin = Location(0, 0)
        distances = []
        for t in range(numTrials):
            f = Field()
            f.addDrunk(Homer, origin)
            distances.append(round(walk(f, Homer,
                                        numTrials), 1))
        return distances

    def drunkTest(walkLengths, numTrials, dClass):
        """Assumes walkLengths a sequence of ints >= 0
             numTrials an int > 0, dClass a subclass of Drunk
           For each number of steps in walkLengths, runs simWalks with
             numTrials walks and prints results"""
        for numSteps in walkLengths:
            distances = simWalks(numSteps, numTrials, dClass)
            print(distances)
            print(dClass.__name__, 'random walk of', numSteps, 'steps')
            print(' Mean =', round(sum(distances)/len(distances), 4))
            print(' Max =', max(distances), 'Min =', min(distances))
            
    #random.seed(0)
    #drunkTest((10, 100, 1000, 10000), 100, UsualDrunk)
    #Pylab
    xVals = [1,2,3,4]
    yVals1 = [1,2,3,4]
    yVals2 = [1,7,3,5]
    plt.plot(xVals, yVals1, 'b-', label = 'first')
    plt.plot(xVals, yVals2, 'r--', label = 'second')
    plt.legend()
    plt.show()
#lecture5()


def lecture6():   
    #Monte Carlo Simulation
    ##A method of estimating the value of an unknown quantity using the principles
    ##of inferential statistics
    #Estimate and the #Confidence
    import random
    class FairRoulette():
        def __init__ (self):
            self.pockets = []

            for i in range(1, 37):
                self.pockets.append(i)
            self.ball = None
            self.pocketOdds = len(self.pockets) - 1

        def spin(self):
            self.ball = random.choice(self.pockets)
        def betPocket(self, pocket, amt):
            if str(pocket) == str(self.ball):
                return amt*self.pocketOdds
            else: return -amt

        def __str__ (self):
            return 'Fair Roulette'

    def playRoulette(game, numSpins, pocket, bet, toPrint):
        totPocket = 0
        for i in range(numSpins):
            game.spin()
            totPocket += game.betPocket(pocket, bet)

        if toPrint:
            
            print('Expected Return Betting of',game, pocket, ' =',\
                  str(100*totPocket/numSpins)+ '%\n')
        return (totPocket/numSpins)

    #The variation of expected return is much larger in case of lower number of spins
    #in case of larger number of spins, the variation of expected returns is much lesser.
    #100 spins are better for ExReturn, but include much variance.
    #Law of Large Numbers, the expected mean goes as closer to the actualMean when the number
    #of samples increase

    #Things do not even out in the future contrary to the poplular belief
    #P(26CR) = 1/2**26  but P(26CR|25CR) = 1/2
    #The exact difference between Gambler Fallacy and Regression to the mean is that
    #Gambler Fallacy would say you will get less than 5 reds in next 10 spins, after extreme 10
    #consecutive reds, but the Regression to the mean would say that in the next 10 spins
    #Number of reds would be lesser than 10 (NOT LESSER THAN 5).


    #Subclass of Fair Roulette
    class EuRoulette(FairRoulette):
        def __init__ (self):
             FairRoulette.__init__(self)
             self.pockets.append('0')

        def __str__ (self):
            return 'EuRoulette'

    #Subclass of European Roulette
    class AmRoulette(FairRoulette):
        def __init__ (self):
            EuRoulette.__init__(self)
            self.pockets.append('00')

        def __str__ (self):
            return 'AmRoulette'


    ##for numSpins in (100,10000000):
    ##    for i in range(1):
    ##        print(numSpins, " of Roulettes")
    ##        playRoulette(FairRoulette(),numSpins, 5, 1, True)
    ##        playRoulette(EuRoulette(), numSpins, 5, 1, True)
    ##        playRoulette(AmRoulette(), numSpins, 5, 1, True)
    ##
        
    #Generally a Fair Roulette is much better for you
    #but less for a casino
    #With Eu roulette and Am not that much better for you
    #but are for casinos


    #X being a list
    def getMeanAndStd(X):
        mean = sum(X)/float(len(X))

        tot = 0.0
        for x in X:
            tot += (x-mean)**2
        std = (tot/len(X))**0.5
        return mean, std


    def findPocketReturn(game, numTrials, trialSize, toPrint):
        pocketReturns = []
        for t in range(numTrials):
            trialVals = playRoulette(game, trialSize, 2, 1, toPrint)
            pocketReturns.append(trialVals)
        return pocketReturns

    random.seed(0)
    numTrials = 20
    resultDict = {}
    games = (FairRoulette, EuRoulette, AmRoulette)
    for G in games:
        resultDict[G().__str__()] = []
    for numSpins in (1000, 10000, 100000, 1000000):
        print('\nSimulate', numTrials, 'trials of',
              numSpins, 'spins each')
        for G in games:
            pocketReturns = findPocketReturn(G(), numTrials,
                                             numSpins, False)
            expReturn = 100*sum(pocketReturns)/len(pocketReturns)
            print('Exp. return for', G(), '=',
                 str(round(expReturn, 4)) + '%')

    #Use Empirical Rule
#lecture6()



def lecture7 ():
    import random, pylab, matplotlib.pyplot as plt, math

    dist, numSamples = [], 100000

    for i in range(numSamples):
        dist.append(random.gauss(0, 100))

    weights = [1/numSamples]*len(dist)
    #in general it is the property of histogram to print y as the number of times the
    #value on the x axis has appeared, weights required to make it normalize
    v = plt.hist(dist, bins = 100, weights = [1/numSamples]*len(dist))
    plt.xlabel('x')
    plt.ylabel('Relative Frequency')
    #print('Fraction withing ~200 of mean = ', sum(v[0][30:70]))
    #plt.show()


    #Checking the empirical rule
    import scipy.integrate

    def gaussian(x, mu, sigma):
        factor1 = (1.0/(sigma*(2*math.pi)**.5))
        factor2 = math.e**-((x-mu)**2/(2*sigma**2))
        return factor1*factor2

    def gaussianChecker():
        xVals, yVals = [], []
        mu, sigma = 0, 1
        x = -4

        while x<=4:
            xVals.append(x)
            yVals.append(gaussian(x, mu, sigma))
            x += 0.5

        plt.plot(xVals, yVals)
        plt.show()

    def checkEmpirical(numTrials):
        for t in range(numTrials):
            mu = random.randint(-10, 10)
            sigma = random.randint(1, 10)
            print("For mu ", mu, " and sigma = ", sigma)
            for numStd in (1, 1.96, 3):
                area = scipy.integrate.quad(gaussian, mu-numStd*sigma, mu+numStd*sigma, (mu, sigma))[0]
                print('Fraction withing ', numStd, " std ", round(area, 4))

    #checkEmpirical(1)
    #print()
    #checkEmpirical(3)
    #Normal Distribution is used in various situations
    #It cannot be used everywhere
    #P(Red) = 0.5 and P(Black) = 0.5 : Uniform Distribution

    #CLT, Samples Mean will be normally distributed
    #For continuous die

    def getMeanAndStd(X):
            mean = sum(X)/float(len(X))

            tot = 0.0
            for x in X:
                tot += (x-mean)**2
            std = (tot/len(X))**0.5
            return mean, std

    def plotMeans(numDice, numRolls, numBins, legend, color, style):
        means = []
        for i in range(numRolls//numDice):
            vals = 0
            for j in range(numDice):
                vals += 5*random.random() 
            means.append(vals/float(numDice))
        plt.hist(means, numBins, color = color, label = legend,
                   weights = [1/len(means)]*len(means),
                   hatch = style)
        return getMeanAndStd(means)


    ##mean, std = plotMeans(1, 1000, 19, '1 die', 'b', '*')
    ##print('Mean of rolling 1 die =', str(mean) + ',', 'Std =', std)
    ##mean, std = plotMeans(50, 1000, 19, 'Mean of 50 dice', 'r', '//')
    ##print('Mean of rolling 50 dice =', str(mean) + ',', 'Std =', std)
    ##plt.title('Rolling Continuous Dice')
    ##plt.xlabel('Value')
    ##plt.ylabel('Probability')
    ##plt.legend()
    ##plt.show()



    class FairRoulette():
            def __init__ (self):
                self.pockets = []

                for i in range(1, 37):
                    self.pockets.append(i)
                self.ball = None
                self.pocketOdds = len(self.pockets) - 1

            def spin(self):
                self.ball = random.choice(self.pockets)
            def betPocket(self, pocket, amt):
                if str(pocket) == str(self.ball):
                    return amt*self.pocketOdds
                else: return -amt

            def __str__ (self):
                return 'Fair Roulette'
    def findPocketReturn(game, numTrials, trialSize, toPrint):
            pocketReturns = []
            for t in range(numTrials):
                trialVals = playRoulette(game, trialSize, 2, 1, toPrint)
                pocketReturns.append(trialVals)
            return pocketReturns
    def playRoulette(game, numSpins, pocket, bet, toPrint):
            totPocket = 0
            for i in range(numSpins):
                game.spin()
                totPocket += game.betPocket(pocket, bet)

            if toPrint:
                
                print('Expected Return Betting of',game, pocket, ' =',\
                      str(100*totPocket/numSpins)+ '%\n')
            return (totPocket/numSpins)

    #Checking of Fair Roulette
    numTrials = 10000
    numSpins = 200
    game= FairRoulette()

    ##means =[]
    ##for i in range(numTrials):
    ##    means.append(findPocketReturn(game, 1, numSpins, False)[0])
    ##
    ##plt.hist(means, bins = 19, weights = [1/len(means)]*len(means))
    ##plt.xlabel('Mean Return')
    ##plt.ylabel('Probability')
    ##plt.show()


    #There is sampling size and numTrials, this needs to be clearly understood
    #The CLT allows us to use the empirical rule when computing confidence intervals
    import statistics
    def throwNeedles(numNeedles):
        inCircle = 0
        for Needles in range(1, numNeedles+1, 1):
            x = random.random()
            y = random.random()

            if (x*x + y*y)**0.5 <= 1.0:
                inCircle += 1

        return 4*(inCircle/float(numNeedles))


    #piValue = throwNeedles(1000)
    #print(piValue)


    def getEst(numNeedles, numTrials):
        estimates = []
        for t in  range(numTrials):
            piGuess = throwNeedles(numNeedles)
            estimates.append(piGuess)
        sDev = statistics.stdev(estimates)
        curEst = sum(estimates)/len(estimates)
        return (curEst, sDev)



    def estPi(precision, numTrials):
        numNeedles = 10
        sDev = precision
        while sDev >= precision/2:
            curEst, sDev = getEst(numNeedles, numTrials)
            print("\nCurrent Estimate ",curEst)
            numNeedles *= 2
        return curEst

    estPi(0.05, 100)
    #Buffon Laplace Method can be used at other place also

#lecture7()



def lecture8():
    #Sampling and Standard Error
    #def lecture8 ():


    import matplotlib.pyplot as plt
    import random, numpy
    #Sampling without replacement

    #First we will prepare data to understand what all this data means
    #Data Munging/ Wrangling/ Preparation

    #print("successUptoThis")
    #cltTestSampleAndPop()
    #Essentialy we can get tighter bounds on the values of means and stds
    #by having a large SampleSize (not necessarilty large numTrials)
    #Then the confidence interval will get tighter and more precised resutls
    #Then essentialy if we have to take a large SampleSize then there is no
    #added benfit of Random Sampling instead of the complete population
    #So we need to deduce better results from a single sample
    #In CLT we consider means of sampleMeans and SDs of sampleMeans



    def makeHist(data, title, xlabel, ylabel, bins = 40):
        plt.hist(data, bins = 20)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        return plt
        

    def getHighs():
        inFile = open('temperatures.csv')
        population = []
        for l in inFile:
            try:
                tempC = float(l.split(',')[1])
                population.append(tempC)
            except:
                continue
        return population


    def getMeansAndSDs(population, sample, verbose = False):
        popMean = sum(population)/len(population)
        sampleMean = sum(sample)/len(sample)
        if verbose:
             makeHist(population,
                     'Daily High 1961-2015, Population\n' +\
                     '(mean = '  + str(round(popMean, 2)) + ')',
                     'Degrees C', 'Number Days')
             
             makeHist(sample, 'Daily High 1961-2015, Sample\n' +\
                     '(mean = ' + str(round(sampleMean, 2)) + ')',
                     'Degrees C', 'Number Days')
             print('Population Mean = ', popMean)
             print('Std of Pop = ', numpy.std(population))

             print('sample mean = ', sampleMean)
             print('Std of Sample = ', numpy.std(sample))
             return popMean, sampleMean, numpy.std(population), numpy.std(sample)
             
    def testSampleAndPop(): 
        random.seed(0)
        population = getHighs()
        for sampleSize in  (100, 200):
            sample = random.sample(population, sampleSize)
            popMean, sampleMean, popSD, sampleSD = getMeansAndSDs(population, sample, True)
            

    #testSampleAndPop()        
    #We can use the 3rd property of Central Limit Theorem
    #Variance of Sample Means 
    def cltTestSampleAndPop():
        random.seed(0)
        population = getHighs()
        sampleSize = 100
        numSamples = 10
        sampleMeans = []

        for i in range(numSamples):
            sample = random.sample(population, sampleSize)
            popMean, sampleMean, popSD, sampleSD = getMeansAndSDs(population, sample, verbose = True)
            sampleMeans.append(sampleMean)
            print('Mean of sample Means = ', round((sum(sampleMeans)/len(sampleMeans)), 3))
            print('SDs of sample sample Means = ', round(numpy.std(sampleMeans), 3))
            makeHist(sampleMeans, 'Means of Sample', 'mean', 'Frequencey')
            plt.axvline(x = popMean, color = 'black')

    #cltTestSampleAndPop()
    #When we start considering the means and SDs of sampleMeans
    #we will eventually get Normal Distribution

    def sem(popSD, sampleSize):
        return popSD/sampleSize**0.5
    #To get information form just 1 sample

    def testSem():
        sampleSizes = (25, 50, 100, 200, 300, 400, 500, 600)
        numTrials = 500
        population = getHighs()
        popSD = numpy.std(population)
        sems = []
        sampleSDs = []
        for size in sampleSizes:
            sems.append(sem(popSD, size))
            means = []
            for t in range(numTrials):
                sample = random.sample(population, size)
                means.append(sum(sample)/len(sample))
            sampleSDs.append(numpy.std(means))
        plt.plot(sampleSizes, sampleSDs, 'r--')
        plt.plot(sampleSizes, sems, 'b-')
        plt.xlabel('SampleSize')
        plt.ylabel('Std and Size')
        plt.show()

    #From one samplSize we need to calculate SEM only once
    #But for SDs we nead a number of trials, SEM of population
    #For SEM calculation we need SDs of Populationm but if dont have it
    #we can calculate the SDs of Sample, for large samples it becomes a good
    #approximation of the SDs of the complete population
    #What matters is skew, if is skewed, we need many samples to
    #get a good approximation of the popSD and sampleSD
    #The popSize generally does not matter, that is why in exit
    #we can consider a small population to make predictions
    #We need independent Random Sample
    #testSem()
        
    def plotDistributions():
        uniform, normal, exp = [], [], []
        for i in range(100000):
            uniform.append(random.random())
            normal.append(random.gauss(0, 1))
            exp.append(random.expovariate(0.5))
        makeHist(uniform, 'Uniform', 'Value', 'Frequency')
        makeHist(normal, 'Gaussian', 'Value', 'Frequency')
        makeHist(exp, 'Exponential', 'Value', 'Frequency')
    #plotDistributions()

    #Are 200 Samples or 200 sampleSize enough ?
    def numBadSamples():
        temps = getHighs()
        popMean = sum(temps)/len(temps)
        sampleSize = 200
        numTrials = 10000
        numBad = 0
        for t in range(numTrials):
            sample = random.sample(temps, sampleSize)
            sampleMean = sum(sample)/sampleSize
            se = numpy.std(sample)/sampleSize**0.5
            if abs(popMean-sampleMean) > 1.96*se:
                numBad += 1
        print('Fraction outside 95% confidence interval = ', numBad/numTrials)


def lecture9():
    #Statistics meets Experimental Data
    import matplotlib.pyplot as plt
    import random, numpy

    def getData(fileName):
        dataFile = open(fileName, 'r')
        distances = []
        masses = []
        dataFile.readline() #reading the first line and discard it
        for line in dataFile:
            d, m = line.split()
            distances.append(float(d))
            masses.append(float(m))
        dataFile.close()
        return (masses, distances)

    def labelPlot():
        plt.title("Measured Displacement")
        plt.xlabel("Force")
        plt.ylabel("Distance")
        
    def plotData(fileName):
        xVals, yVals = getData(fileName)
        #The benefit of having a numpy.array is we
        #can use Math functions without calling the loops
        xVals = numpy.array(xVals)
        yVals = numpy.array(yVals)
        xVals *= 9.81 #Converting to Weight from Mass
        plt.plot(xVals, yVals, 'bo', label = 'Measured Displacement')
        labelPlot()
        plt.show()
    #plotData('experiment.txt')
        
    def fitData(fileName):
        xVals, yVals = getData(fileName)
        xVals = numpy.array(xVals)
        yVals = numpy.array(yVals)
        xVals = xVals*9.81 #get force
        plt.plot(xVals, yVals, 'bo', label = 'measured points')
        labelPlot()
        a, b = numpy.polyfit(xVals, yVals, 1)
        estYvals = a*numpy.array(xVals) + b
        print("a= ",a," b= ", b)
        plt.plot(xVals, estYvals, 'yellow', label = 'Linear Fit, k = '+ str(round(1/a, 5)))
        plt.legend()
    #fitData('experiment.txt')

    ##def fitData2(fileName):
    ##    xVals, yVals = getData(fileName)
    ##    xVals = numpy.array(xVals)
    ##    yVals = numpy.array(yVals)
    ##    xVals = xVals*9.81 #get force
    ##    plt.plot(xVals, yVals, 'bo', label = 'measured points')
    ##    labelPlot()
    ##    a, b = numpy.polyfit(xVals, yVals, 1)
    ##    estYvals = a*numpy.array(xVals) + b
    ##    print("a= ",a," b= ", b)
    ##    plt.plot(xVals, estYvals, 'black', label = 'Linear Fit, k = '+ str(round(1/a, 5)))
    ##    plt.legend()
    ##    #plt.show()
    ##
    ##fitData2('experiment.txt')

    def fitData1(filename, degree):
        xVals, yVals = getData('experiment.txt')
        xVals = numpy.array(xVals)
        yVals = numpy.array(yVals)
        xVals *= 9.81 #To get Force from mass values
        plt.plot(xVals, yVals, 'bo', label = 'Measured Points')
        labelPlot()

        #Now we start fitting curves instead of lines
        model = numpy.polyfit(xVals, yVals, degree)
        estYVals = numpy.polyval(model, xVals)
        print("The Coefficients are ", model)
        plt.plot(xVals, estYVals, 'red')
        plt.show()
    #fitData1('experiment.txt', 2)


    #It is square of diff and not the mode
    #So that it attains one and only one minima
    #in the given space

    #How good are these fits, so we need a param function
    #instead of looking at the data, relative fit
    def aveMeanSquareError(data, predicted):
        error = 0.0
        for i in range(len(data)):
            error += (data[i]-predicted[i])**2
        return error/len(data)

    ##def compareTwoModels():
    ##    estYVals = numpy.polyval(model1, xVals)  
    ##    print('Ave. mean square error for linear model =',
    ##          aveMeanSquareError(yVals, estYVals))
    ##    estYVals = numpy.polyval(model2, xVals)
    ##    print('Ave. mean square error for quadratic model =',
    ##          aveMeanSquareError(yVals, estYVals))

    def rSquared(observed, predicted):
        error = ((predicted - observed)**2).sum()
        meanError = error/len(observed)
        return 1-(meanError/numpy.var(observed))
    #In abosulte sense, more the value of R**2
    #in general the better explanation of variability is we get


    def genFits(xVals, yVals, degrees):
        models = []
        for d in degrees:
            model = numpy.polyfit(xVals, yVals, d)
            models.append(model)
        return models

    def testFits(models, degrees, xVals, yVals, title):
        plt.plot(xVals, yVals, 'o', label = 'Data')
        for i in range(len(models)):
            estYVals = numpy.polyval(models[i], xVals)
            error = rSquared(yVals, estYVals)
            plt.plot(xVals, estYVals,
                       label = 'Fit of degree '\
                       + str(degrees[i])\
                       + ', R2 = ' + str(round(error, 5)))
        plt.legend(loc = 'best')
        plt.title(title)
        plt.show()

    xVals, yVals = getData('experiment.txt')
    xVals = numpy.array(xVals)
    yVals = numpy.array(yVals)
    xVals *= 9.81 #To get Force from mass values

    degrees = tuple([i for i in range(10)])
    models = genFits(xVals, yVals, degrees)
    testFits(models, degrees, xVals, yVals, "Comaprison")
    #We can get much tighter bounds by increasing the degree
    #of the fitting polynomial, but this model does not scale well
    #at certaing poitns, it becomes too dependent and scalability reduces


def lecture10():
    #Statistics meets Experimental Data
    import matplotlib.pyplot as plt
    import numpy, pandas, random
    def genNoisyParabolicData(a, b, c, xVals, fName):
        yVals = []
        for x in xVals:
            theorticalVal = a*x**2 + b*x + c
            yVals.append(theorticalVal+random.gauss(0, 35))
        f = open(fName, "w")
        f.write('x       y\n')
        for i in range(len(yVals)):
            f.write(str(yVals[i]) + ',' +str(xVals[i]) + '\n')
        f.close()
        
    xVals = range(-10, 11, 1)
    a, b, c = 3.0, 0.0, 0.0

    random.seed(0)
    genNoisyParabolicData(a, b, c, xVals, 'Dataset1.txt')
    genNoisyParabolicData(a, b, c, xVals, 'Dataset2.txt')
    #Data Generations is complete

    #Model Testing
    def rSquared(observed, predicted):
            error = ((predicted - observed)**2).sum()
            meanError = error/len(observed)
            return 1-(meanError/numpy.var(observed))
        
    def getData(fileName):
            dataFile = open(fileName, 'r')
            distances = []
            masses = []
            dataFile.readline() #reading the first line and discard it
            for line in dataFile:
                d, m = line.split(',')
                distances.append(float(d))
                masses.append(float(m))
            dataFile.close()
            return (masses, distances)

    def genFits(xVals, yVals, degrees):
            models = []
            for d in degrees:
                model = numpy.polyfit(xVals, yVals, d)
                models.append(model)
            return models

    def testFits(models, degrees, xVals, yVals, title):
            plt.plot(xVals, yVals, 'o', label = 'Data')
            for i in range(len(models)):
                estYVals = numpy.polyval(models[i], xVals)
                error = rSquared(yVals, estYVals)
                plt.plot(xVals, estYVals,
                           label = 'Fit of degree '\
                           + str(degrees[i])\
                           + ', R2 = ' + str(round(error, 5)))
            plt.legend(loc = 'best')
            plt.title(title)
            
            
    degrees = (2, 3, 4, 8, 16)
    xVals1, yVals1 = getData('Dataset1.txt')
    models1 = genFits(xVals1, yVals1, degrees)
    testFits(models1, degrees, xVals1, yVals1, 'Dataset1.txt')
    plt.figure()
    xVals2, yVals2 = getData('Dataset2.txt')
    models2 = genFits(xVals2, yVals2, degrees)
    testFits(models2, degrees, xVals2, yVals2, 'Dataset2.txt')
    #The overfitting that we just saw is generated from the training error
    #Small training error a necessary condition for a great, scalable model
    #We want model to work well on other data generated by the same process

    #Cross Validate Use models for Dataset 1 to predict Dataset2 and vice versa
    plt.figure()
    testFits(models1, degrees, xVals2, yVals2, 'DataSet 2/Model 1')
    plt.figure()
    testFits(models2, degrees, xVals1, yVals1, 'Dataset 1/Model 2')
    plt.show()
    #From this experimental plot, we see degree 2-4 are quite well fitting
    #not only on the training data but also on the testing data
    #Overfitting -- Fitting the underlying process and the internal noise 
    #For linear data predictive ability of first order fit
    #is much better than second order fit.
    #Our objective should be Balancing Fit with Complexity


    #Leave out one cross Validation -- For small dataset
    #leave one data out and create model using other vals, and test on left out value
    #Similarly for other values in the dataset

    #K fold Cross Validation
    #instead of leaving one out, just leave k sized group out
    #later use for testing

    #Random sampling Validation
    #Selecting some random n values (20%-30%)


    #Doing this complete predictionality

    #def lectureFittingTemperature():
    def rSquared(observed, predicted):
        error = ((predicted - observed)**2).sum()
        meanError = error/len(observed)
        return 1 - (meanError/numpy.var(observed))

    def genFits(xVals, yVals, degrees):
        models = []
        for d in degrees:
            model = numpy.polyfit(xVals, yVals, d)
            models.append(model)
        return models

    def testFits(models, degrees, xVals, yVals, title):
        plt.plot(xVals, yVals, 'o', label = 'Data')
        for i in range(len(models)):
            estYVals = plt.polyval(models[i], xVals)
            error = rSquared(yVals, estYVals)
            plt.plot(xVals, estYVals,
                       label = 'Fit of degree '\
                       + str(degrees[i])\
                       + ', R2 = ' + str(round(error, 5)))
        plt.legend(loc = 'best')
        plt.title(title)

    def getData(fileName):
        dataFile = open(fileName, 'r')
        distances = []
        masses = []
        dataFile.readline() #discard header
        for line in dataFile:
            d, m = line.split()
            distances.append(float(d))
            masses.append(float(m))
        dataFile.close()
        return (masses, distances)

    def labelPlot():
        pylab.title('Measured Displacement of Spring')
        pylab.xlabel('|Force| (Newtons)')
        pylab.ylabel('Distance (meters)')

    def plotData(fileName):
        xVals, yVals = getData(fileName)
        xVals = pylab.array(xVals)
        yVals = pylab.array(yVals)
        xVals = xVals*9.81  #acc. due to gravity
        pylab.plot(xVals, yVals, 'bo',
                   label = 'Measured displacements')
        labelPlot()

    def fitData(fileName):
        xVals, yVals = getData(fileName)
        xVals = pylab.array(xVals)
        yVals = pylab.array(yVals)
        xVals = xVals*9.81 #get force
        pylab.plot(xVals, yVals, 'bo',
                   label = 'Measured points')                 
        model = pylab.polyfit(xVals, yVals, 1)
        xVals = xVals + [2]
        yVals = yVals + []
        estYVals = pylab.polyval(model, xVals)
        pylab.plot(xVals, estYVals, 'r',
                   label = 'Linear fit, r**2 = '
                   + str(round(rSquared(yVals, estYVals), 5)))                
        model = pylab.polyfit(xVals, yVals, 2)
        estYVals = pylab.polyval(model, xVals)
        pylab.plot(xVals, estYVals, 'g--',
                   label = 'Quadratic fit, r**2 = '
                   + str(round(rSquared(yVals, estYVals), 5)))
        pylab.title('A Linear Spring')
        labelPlot()
        pylab.legend(loc = 'best')

    random.seed(0)

    class tempDatum(object):
        def __init__ (self, s):
            info = s.split(',')
            self.high = info[1]
            self.year = int(info[2][0:4])

        def getHigh(self):
            return self.high

        def getYear(self):
            return self.year

    def getTempData():
        inFile = open('temperatures.csv')
        data  = []
        for l in inFile:
            data.append(tempDatum(l))
        return data

    def getYearlyMeans(data):
        years = {}
        for d in data:
            try:
                years[d.getYear()].append(d.getHigh())
            except:
                years[d.getYear()] = [d.getHigh()]
        for y in years:
            years[y] = sum(years[y])/len(years[y])
        return years

    data =  getTempData()
    years= getYearlyMeans(data)
    xVals, yVals = [], []
    for e in years:
        xVals.append(e)
        yVals.append(years[e])
    plt.plot(xVals, yVals)
    plt.xlabel('Year')
    plt.ylabel('Mean daily high temperature')

    def splitData(xVals, yVals):
        toTrain = random.sample(range(len(xVals)), len(xVals)//2)
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(xVals)):
            if i in toTrain:
                trainX.append(xVals[i])
                trainY.append(yVals[i])
            else:
                testX.append(xVals[i])
                testY.append(yVals[i])
        return trainX, trainY, testX, testY

    numSubsets = 10
    dimensions = (1, 2, 3, 4)
    rSquares = {}
    for d in dimensions:
        rSquares[d] = []
            
    for f in range(numSubsets):
        trainX, trainY, testX, testY = splitData(xVals, yVals)
        for d in dimensions:
            model = numpy.polyfit(trainX, trainY, d)
            estYVals = nunpy.polyval(model, trainX)
            estYVals = numpy.polyval(model, testX)
            rSquares[d].append(rSquared(testY, estYVals))
    print('Mean R-squares for test data')
    for d in dimensions:
        mean = round(sum(rSquares[d])/len(rSquares[d]), 4)
        sd = round(numpy.std(rSquares[d]), 4)
        print('For dimensionality', d, 'mean =', mean,
              'Std =', sd)
    print(rSquares[1])



#The Machine Learning 101 MITOCW
def lecture11():
    import matplotlib.pyplot as plt
    import random, numpy, pandas

    def variance(X):
        mean = float(sum(X))/len(X)
        diffs = 0.0
        for x in X:
            diffs += (x-mean)**2
        return diffs/len(X)

    def stdDev(X):
        return variance**0.5

    def minkowskiDist(v1, v2, p):
        dist = 0.0
        for i in range(len(v1)):
            dist += abs(v1[i]-v2[i])**p
        return dist**(1.0/p)


    class Animal(object):
        def __init__(self, name, features):
            self.name = name
            self.features = numpy.array(features)

        def getName(self):
            return self.name

        def getFeatures(self):
            return self.features

        #We need these feature vector distance to get information about similarity
        def distance(self, other):
            return minkowskiDist(self.getFeatures(), other.getFeatures(), 2)

        def __str__ (self):
            return (self.name + " and " + self.features)

    def compareAnimals(animals, precision):
        """"Builds a table of Euclidean distance between each animal"""
        columnLabels = []
        for a in animals:
            columnLabels.append(a.getName())
        rowLabels = columnLabels[:]
        tableVals = []
        #Get distance between pairs of animals
        for a1 in animals:
            row = []
            for a2 in animals:
                if a1==a2:
                    row.append('--')
                else:
                    distance = a1.distance(a2)
                    row.append(str(round(distance, precision)))
            tableVals.append(row)
        #Produce Table
        table = plt.table(rowLabels = rowLabels, colLabels = columnLabels,\
                          cellText = tableVals, cellLoc= 'center', loc = 'center',\
                          colWidths = [0.2]*len(animals))
        table.scale(1, 2.5)
        plt.title("Euclidean Distances")
        


    rattlesnake = Animal('rattlesnake', [1,1,1,1,0])
    boa = Animal('boa\nconstrictor', [0,1,0,1,0])
    dartFrog = Animal('dart frog', [1,0,1,0,4])
    ##alligator = Animal('alligator', [1,1,0,1,4])
    ##Here the "legs" dimension is particularly too disproportionately large
    ##
    ##animals = [rattlesnake, boa, dartFrog, alligator]
    ##compareAnimals(animals, 3)
    alligator = Animal('alligator', [1,1,0,1,1])
    #That is why using Binary Feature is helpful
    animals = [rattlesnake, boa, dartFrog, alligator]
    #Feature Engineering Matters
    #When given UNLABELED data, try to find clusters of examples near each other
    #Use centroids of clusters as definition of each leaned class
    #New data assigned to closest cluster (as measured from centroid of class)
    #When given labeled data, lean mathematical surfaces that "best" separates
    #labeled examples, subject to constraints on complexity (Avoid Overfitting)

    compareAnimals(animals, 3)
    plt.show()
   



#Clustering
#def lecture12():

















