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
