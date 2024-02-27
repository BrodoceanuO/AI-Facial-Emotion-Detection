import math
from itertools import groupby
import datetime
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

class EmotionGraph: #class created for reusing the emotion file and X/Y value data
    def __init__(self):
        self.filepath = None
        self.X_time = None
        self.Y_emotion = None
        self.domin_emot = []

    def readFromFile(self, filename):
        self.filepath = filename
        file = None
        try:
            file = open(filename, "r")
        except FileNotFoundError:
            print("The specified file was not found")

        X_time = []
        Y_emotion = []

        lines = file.readlines()
        for line in lines[1:]:
            split_line = line.split(" ")
            X_time.append(float(split_line[0]))
            Y_emotion.append(float(split_line[1]))
            self.domin_emot.append(int(split_line[2]))

        self.X_time = np.array(X_time)
        self.Y_emotion = np.array(Y_emotion)


    def plotGraph(self):
        plt.ylim([-1,1])
        timeplot, = plt.plot(self.X_time, self.Y_emotion, color="blue")
        plt.show()

    def calcEngagement(self):
        c = Counter(self.Y_emotion)
        not_neutral = len(self.Y_emotion) - c[float(0)]
        engagement = not_neutral * 100 / len(self.Y_emotion)
        print(f"Engagement: {math.floor(engagement)}%")
        return engagement

    def calcPosPercetage(self):

        pos = np.where(self.Y_emotion > 0)[0]

        pos_percentage = len(pos) * 100 / len(self.Y_emotion)

        print("Positive percentage: %.2f%%" % pos_percentage)

        return pos_percentage

    def calcNegPercetage(self):

        neg = np.where(self.Y_emotion < 0)[0]

        neg_percentage = len(neg) * 100 / len(self.Y_emotion)

        print("Negative percentage: %.2f%%" % neg_percentage)

        return neg_percentage

    def calcNeutralPercentage(self):

        neutral = np.where(self.Y_emotion == float(0))[0]

        neutral_percentage = len(neutral) * 100 / len(self.Y_emotion)

        print("Neutral percentage: %.2f%%" % neutral_percentage)

        return neutral_percentage

    def calcMean(self):

        mean = np.mean(self.Y_emotion)

        if mean > 0:
            print("Mean is: %.2f The crowd was positively engaged" % mean)
        elif mean < 0:
            print("Mean is: %.2f The crowd was negatively engaged" % mean)
        elif mean == 0:
            print("Mean is: %.2f The crowd was not engaged" % mean)

        return mean

    def calcMaxPosEngagementTime(self):
        g = groupby(self.Y_emotion, key=lambda x: x > 0.0)
        m = max([list(s) for v, s in g if v > 0.0], key=len)
        first_index = np.where(self.Y_emotion == m[0])[0][0]
        last_index = np.where(self.Y_emotion == m[-1])[0][0]
        period = int(self.X_time[last_index] - self.X_time[first_index])
        time_formatted = str(datetime.timedelta(seconds=period))
        start_time = str(datetime.timedelta(seconds=self.X_time[first_index]))
        end_time = str(datetime.timedelta(seconds=self.X_time[last_index]))
        print("The longest period the crowd was positively engaged for was %s during %s to %s" % (time_formatted, start_time, end_time))

    def calcMaxNegEngagementTime(self):
        g = groupby(self.Y_emotion, key=lambda x: x < 0.0)
        m = max([list(s) for v, s in g if v > 0.0], key=len)
        first_index = np.where(self.Y_emotion == m[0])[0][0]
        last_index = np.where(self.Y_emotion == m[-1])[0][0]
        period = int(self.X_time[last_index] - self.X_time[first_index])
        time_formatted = str(datetime.timedelta(seconds=period))
        start_time = str(datetime.timedelta(seconds=self.X_time[first_index]))
        end_time = str(datetime.timedelta(seconds=self.X_time[last_index]))
        print("The longest period the crowd was negatively engaged for was %s during %s to %s" % (time_formatted, start_time, end_time))

    def calcMaxEngagementTime(self):
        g = groupby(self.Y_emotion, key=lambda x: x != 0.0)
        m = max([list(s) for v, s in g if v > 0.0], key=len)
        first_index = np.where(self.Y_emotion == m[0])[0][0]
        last_index = np.where(self.Y_emotion == m[-1])[0][0]
        period = int(self.X_time[last_index] - self.X_time[first_index])
        time_formatted = str(datetime.timedelta(seconds=period))
        print("The longest period the crowd was engaged for was %s" % time_formatted)

    def calcMaxPos(self):
        maxPositive = max(self.Y_emotion)
        index = np.where(self.Y_emotion == maxPositive)[0][0]
        maxTime = self.X_time[index]
        time_formatted = str(datetime.timedelta(seconds=maxTime))
        print("The most positive reaction out of the crowd was %s at %s" % (maxPositive, time_formatted))

    def calcMaxNeg(self):
        maxNegative = min(self.Y_emotion)
        index = np.where(self.Y_emotion == maxNegative)[0][0]
        maxTime = self.X_time[index]
        time_formatted = str(datetime.timedelta(seconds=maxTime))
        print("The most negative reaction out of the crowd was %s at %s" % (maxNegative, time_formatted))

    def getGraphForVid(self, timestamp):
        X_index = np.where(self.X_time < timestamp)
        X_slice = self.X_time[X_index]
        Y_slice = self.Y_emotion[:len(X_slice)]
        return X_slice, Y_slice

    def printEmotionsByOrder(self):
        c = Counter(self.domin_emot)
        print(c.most_common())
        most_comm = c.most_common()
        print("The predominant emotions experienced by the crowd ordered by number of appearances is:")
        for i in range(len(most_comm)):
            if mapping[most_comm[i][0]] == "Mixed":
                continue
            print(mapping[most_comm[i][0]])

    def calcAllStats(self):
        self.calcEngagement()
        self.calcPosPercetage()
        self.calcNegPercetage()
        self.calcNeutralPercentage()
        self.calcMean()
        self.calcMaxPosEngagementTime()
        self.calcMaxNegEngagementTime()
        self.calcMaxEngagementTime()
        self.calcMaxPos()
        self.calcMaxNeg()
