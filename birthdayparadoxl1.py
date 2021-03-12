def birthdayparadox():
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

birthdayparadox()
