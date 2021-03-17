#def lecture14():
#How to lie with Stats, it is always a better idea to
#VISUALIZE data, the scale is also important
#Statistics about Data and Data are different things
#Things being compared are actually comparable or not
#GIGO: Garbage In, Garbage Out, always choos GOOD data
#Non-Response Bias, Do not draw conclusions
#Always it is important to understand how data is collected.
#Fluctuations and Trends are different, find good interval
#We cannot draw conclusion about interval by looking at the boundaries only
#Using percentage understand the complete stats of the BASIC Set
#Context Matters most importantly, put context on a number
#Percentage Change with context matters not the Percentage Change 
#The data might have bad luck, random data variation
#Avoid Cherry Picking in the data (Multiple Hypothesus Testing)
#Texas Sharpshooter Fallacy

import random
import matplotlib.pyplot as plt

random.seed(0)
numCasesPerYear = 36000
numYears = 3
stateSize = 10000
communitySize = 10
numCommunities = stateSize//communitySize

numTrials = 100
numGreater = 0
for t in range(numTrials):
    locs = [0]*numCommunities
    for i in range(numYears*numCasesPerYear):
        locs[random.choice(range(numCommunities))] += 1
    if locs[111] >= 143:
        numGreater += 1
prob = round(numGreater/numTrials, 4)
print('Est. probability of region 111 having\
 at least 143 cases =', prob)


numTrials = 100
anyRegion = 0
for trial in range(numTrials):
    locs = [0]*numCommunities
    for i in range(numYears*numCasesPerYear):
        locs[random.choice(range(numCommunities))] += 1
    if max(locs) >= 143:
        anyRegion += 1
print(anyRegion)
aProb = round(anyRegion/numTrials, 4)
print('Est. probability of some region  having\
 at least 143 cases =', aProb)
