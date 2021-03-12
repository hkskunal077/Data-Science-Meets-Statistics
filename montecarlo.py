def montecarlo():   
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

montecarlo()
