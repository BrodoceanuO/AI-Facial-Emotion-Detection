import cv2  # pip install opencv-python
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from collections import Counter
from emotiongraph import EmotionGraph
import utils


import warnings
warnings.filterwarnings("ignore") #added for presentation purposes,
                                  #diregards deprecation warnings for some of the functions used

path = "FinalModel.h5"  #path to the model

mapping = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Fear"}
#angry - negative, happy - positive, neutral - neutral, fearful - negative

def main():
    print("The emotion mapping used is the following: angry - negative, happy - positive, neutral - neutral, fearful - negative")
    while True:
        key = input("Enter option: 1 for recording, 2 for plotting an emotion graph, press q to quit\n")
        if key == "1":
            utils.emotion_graph(mapping, path, record_graph=True)
        elif key == "2":
            graph_path = input("Enter the path of the graph file:\n")

            eg = EmotionGraph()
            eg.readFromFile(graph_path)

            while True:
                gkey = input("Enter the stats you would like to view, type L to list the available options, q to quit viewing graph mode\n")
                if gkey == "L":
                    print("1 - plot Graph")
                    print("2 - show all stats")
                    print("3 - show engagement percentage")
                    print("4 - show positive engagement percentage")
                    print("5 - show negative engagement percentage")
                    print("6 - show neutral percentage")
                    print("7 - show mean")
                    print("8 - show maximum positive enagement time")
                    print("9 - show maximum negative engagement time")
                    print("10 - show maximum engagement time, regardless of classification")
                    print("11 - show maximum y value")
                    print("12 - show minimum y value")
                elif gkey == "1":
                    eg.plotGraph()
                elif gkey == "2":
                    eg.calcAllStats()
                elif gkey == "3":
                    eg.calcEngagement()
                elif gkey == "4":
                    eg.calcPosPercetage()
                elif gkey == "5":
                    eg.calcNegPercetage()
                elif gkey == "6":
                    eg.calcNeutralPercentage()
                elif gkey == "7":
                    eg.calcMean()
                elif gkey == "8":
                    eg.calcMaxPosEngagementTime()
                elif gkey == "9":
                    eg.calcMaxNegEngagementTime()
                elif gkey == "10":
                    eg.calcMaxEngagementTime()
                elif gkey == "11":
                    eg.calcMaxPos()
                elif gkey == "12":
                    eg.calcMaxNeg()
                elif gkey == "q":
                    break
                else:
                    print("The key you typed is invalid")

        elif key == "q":
            break
        else:
            print("The key you typed is invalid")

if __name__ == "__main__":
    main()