def randomwalk():

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

randomwalk()
