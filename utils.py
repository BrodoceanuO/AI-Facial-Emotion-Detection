import cv2  # pip install opencv-python
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from collections import Counter
from emotiongraph import EmotionGraph


import warnings
warnings.filterwarnings("ignore") #added for presentation purposes,
                                  #diregards deprecation warnings for some of the functions used

path = "FinalModel.h5"  #path to the model

mapping = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Fear"}
#angry - negative, happy - positive, neutral - neutral, fearful - negative


def testEmotionGraph():
    eg = EmotionGraph()
    eg.readFromFile("03_Jun_2022_11h_11min_40s_GMT.txt")
    eg.calcEngagement()
    eg.calcPosPercetage()
    eg.calcNegPercetage()
    eg.calcNeutralPercentage()
    eg.calcMean()
    eg.calcMaxPosEngagementTime()
    eg.calcMaxNegEngagementTime()
    eg.calcMaxEngagementTime()
    eg.calcMaxPos()
    eg.calcMaxNeg()


# def getMaxElement(queue):
#     c = Counter(queue)
#     max_element = c.most_common()
#     return max_element[0][0]

def changeY(stat, pred): #function returning the y value using the prediction % and emotion type
    values = {0 : -1, 1 : 1, 2 : 0, 3 : -1} #angry - negative, happy - positive, neutral - neutral, fearful - negative
    return pred*values[stat]

def getMean(stat_list, stat_values):
    sum = 0
    for i in range(len(stat_list)):
        sum += changeY(stat_list[i], stat_values[i])
    return sum/len(stat_list)

def getMajority(stat_list):
    c = Counter(stat_list)
    #print(c.most_common())
    most_comm = c.most_common()
    print(most_comm)
    if len(stat_list) > 1 and len(most_comm) == len(stat_list):
        emotion = 2
        count = -1
        return emotion, count
    emotion, count = most_comm[0]
    return emotion, count

def webcam_emotion(new_model, mapping, record_graph = False):
    file = None
    if record_graph:
        init_time = time.localtime()
        filename = time.strftime('%d_%b_%Y_%Hh_%Mmin_%Ss_GMT.txt', init_time)
        file = open('graphs/' + filename,"w")
        file.write(time.strftime('%d_%b_%Y_%Hh_%Mmin_%Ss_GMT\n', init_time))
        print("The graph for this recording is being saved to " + str(filename))
    dim = (480, 640)
    plot_dims = (250, 250)

    paddown = (dim[0] - plot_dims[0])
    padleft = (dim[1] - plot_dims[1])

    plt.style.use('ggplot')
    plt.ylim([-1,1])

    fig = plt.figure()
    stat_list = []
    stat_values = []

    X_time = []

    Y_emotion = []

    y_value = 0

    print("opening webcam...")
    cap = cv2.VideoCapture(0)
    # continue only if webcam is opened
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Webcam cannot be opened")
    print("webcam opened")
    start = time.time()
    while True:
        ret, frame = cap.read()
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        emotion = "-1"
        if len(faces) == 0:
            y_value = 0
            emotion = "-1"  # no faces detected
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("Face not detected")
                emotion = 2
            else:
                for (ex, ey, ew, eh) in facess: # repeat for all faces in image
                    face_roi = roi_color[ey: ey + eh, ex:ex + ew]  # getting the face
                    final_image = cv2.resize(face_roi, (48, 48)) # resizing to model input
                    final_image = np.expand_dims(final_image, axis=0)  # model needs extra dimension
                    final_image = final_image / 255.0 #normalizing
                    Predictions = new_model.predict(final_image) #getting prediction % for each class
                    stat = np.argmax(Predictions) #getting the class given by the model's prediction
                    stat_value = np.amax(Predictions) #getting the % attributed to the prediction

                    stat_list.append(stat) #save results for all faces in image
                    stat_values.append(stat_value)

                    font = cv2.FONT_HERSHEY_PLAIN
                    printEmotion(frame, mapping, stat, x, y, w, h, font)

        if len(stat_list) > 0:
            y_value = getMean(stat_list, stat_values)
            #print("stat list is" + str(stat_list))
            emotion, count = getMajority(stat_list)
            #emotion = emotion[0]
            printMajorityEmotion(frame, mapping, emotion, count, len(stat_list))
        curr_time = time.time()


        #(stat_list)

        stat_list = [] #clean list for use in the next cycle
        stat_values = []

        X_time.append(curr_time - start)
        Y_emotion.append(y_value)

        if file:
            file.write(str(curr_time - start) + " " + str(y_value) + " " + str(emotion) + "\n") #write to file if it exists

        plt.ylim([-1, 1])
        timeplot, = plt.plot(X_time[-200:], Y_emotion[-200:], color = "blue") #plotting the last 200 values for x and y in real time

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, plot_dims, interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.copyMakeBorder(img, 0, paddown, padleft, 0, cv2.BORDER_CONSTANT, None, value= 0)


        cv2.rectangle(frame, (640-plot_dims[0], 0), (640, plot_dims[1] - 10), (0, 0, 0), -1)
        dst = cv2.addWeighted(frame, 1, img, 1, 0) #adding the graph over the webcam image

        #stat = np.argmax(Predictions)

        cv2.imshow('FER mobilenet', dst)
        if cv2.waitKey(1) & 0xFF == 27: # 27 means the escape key
            break
    cap.release()
    cv2.destroyAllWindows()
    file.close()
    end = time.time()

def printEmotion(frame, mapping, stat, x, y, w, h, font):

    status = mapping[stat]
    cv2.putText(frame, status, (x-100, y+w//2), font, 2, (0, 0, 255), 2, cv2.LINE_4)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

def printMajorityEmotion(frame, mapping, stat, count, len_faces):

    #print("Count is: " + str(count))
    #print("Len Faces is: " + str(len_faces))
    if count == -1:
        status = "Majority: " + mapping[stat] + "(mixed)"
    else:
        status = "Majority: " + mapping[stat] + " (" + str(count*100//len_faces) + "%)"

    x1,y1,w1,h1 = 100,0,175,75

    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


def emotion_graph(mapping, path, record_graph = False):

    new_model = tf.keras.models.load_model(path)
    start = time.time()
    webcam_emotion(new_model, mapping, record_graph = record_graph)
    end = time.time()

    time_consumed=end-start

    print(time_consumed)