import xport
import math
import matplotlib.pyplot as plt
import numpy as np

__DEBUG__ = False

INDIV_FOODS_SEQN_INDEX = 0
INDIV_FOODS_TIME_INDEX = 13
INDIV_FOODS_IRON_INDEX = 55

TOTAL_HOURS = 24
SEC_PER_HR = 3600

def parseDataset(fileName):
    '''
    NHANES Dietary Data - Dietary Interview - Individual Foods
    '''
    
    if __DEBUG__: count = 0
    
    participants = {} # seqn : [iron intake at each hour of the day]
    with open(fileName, 'rb') as f:
        for row in xport.Reader(f):
            seqn = int(row[INDIV_FOODS_SEQN_INDEX])
            time = int(math.floor(row[INDIV_FOODS_TIME_INDEX] / SEC_PER_HR))
            iron = row[INDIV_FOODS_IRON_INDEX]
            if math.isnan(iron):
                continue

            if seqn not in participants:
                participants[seqn] = [0] * TOTAL_HOURS
                participants[seqn][time] = iron
            else:
                participants[seqn][time] += iron

            if __DEBUG__:
                count += 1
                if count > 50:
                    break

    return participants

def processDataset(participants):

    allIronIntakes = np.array(list(participants.values()))
    seqn = np.array(list(participants.keys())).reshape(len(participants), 1)
    print(seqn.shape)
    rowSums = np.sum(allIronIntakes, axis=1, keepdims=True)
    print((rowSums == 0).shape)
    seqn[rowSums == 0] = -1
    rowSums[rowSums == 0] = 1 # avoid dividing by 0 if participant has no iron intake

    normalizedData = allIronIntakes / rowSums
    normalizedData = normalizedData[~np.all(normalizedData == 0, axis = 1)]
    seqn = seqn[~np.all(seqn == -1, axis = 1)]
    
    metadata = {}    
    for i, data in enumerate(normalizedData):
        metadata[tuple(data)] = [seqn[i], rowSums[i]] 
    print(rowSums)
    return normalizedData, metadata

def plotIronIntake(title, ironIntake, imgFileName):
    '''
    title: participant's sequence number or centroid index
    ironIntake: [iron intake at ith hour] * 24
    imgFileName: name of output image file
    '''

    if __DEBUG__:
        print("Plotting " + title + "'s iron intake...")
    plt.figure()
    plt.xlim([0, TOTAL_HOURS])
    plt.xticks(range(TOTAL_HOURS), range(TOTAL_HOURS))
    plt.xlabel("Time (hours)")
    plt.ylabel("% Iron Intake")
    plt.title(title)
    hr = range(24)
    plt.plot(hr, ironIntake, '.-', label=title)
    plt.savefig(imgFileName)


