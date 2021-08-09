import cv2
import numpy as np
from depthai_helpers.utils import to_tensor_result
import pyttsx3
import speech_recognition as sr
r = sr.Recognizer()
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28],
          [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

text = 'Welcome to palpose choose the mode what you want" " '
engine = pyttsx3.init()
engine.setProperty("rate", 155)
engine.say(text)

engine.runAndWait()
text2 = ' if you want sitting tracking say sitting  ' \
        ', and to have   standing TRACKING  say  standing '\

engine.say(text2)
engine.runAndWait()
text3=' And if you want  to diagnose bowlegs or Kyphosis or Scoliosis  Say  it   '
engine.say(text3)
engine.runAndWait()
def writeText (frame,txt,color,position):
    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (400, 300)
    fontScale = 0.5

    fontColor = color
    lineType = 2

    cv2.putText(frame, txt,
                position,
                font,
                fontScale,
                fontColor,
                lineType)
def  RightOrLeftSide(list,p):
    leftscore = 10
    rightscore = 10
    for i in range(len(left)):
        if (list[left[i] + (18 * p)][2] == -1):
            leftscore = leftscore - 1
        if (list[right[i] + (18 * p)][2] == -1):
            rightscore = rightscore - 1
            return leftscore, rightscore
def IsSettingOrStanding(list, p) :
    leftscore, rightscore = RightOrLeftSide(list, p)
    if (leftscore > rightscore):
        if ((list[12 + (18 * p)][1] - list[11 + (18 * p)][1] ) >= 40 ):
            return "standing"
        else:
            return "sitting"

    else:
        if ((list[9 + (18 * p)][1] - list[8+ (18 * p)][1]) >= 40):
            return "standing"
        else:
            return "sitting"


def detectSitting(nn_manager,list,p) :
      #print(list)
      score = list
      #print("len(list)")
      #print(len(list))

      w, h = nn_manager.input_size
      w2=float(w)
      try :
         leftscore, rightscore = RightOrLeftSide(list,p)

         if (leftscore > rightscore) :
           print("left")

           if ((list[11 + (18 * p)][0]==-1 )):
               if ((list[17 + (18 * p)][0] - list[5 + (18 * p)][0]) / w2 <-0.025 or(list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 <-0.17):
                   return "head forward"
               else:
                   return "normal head alignment"

           elif (((list[17+(18*p)][0] - list[11+(18*p)][0])/w2 < -0.055 )):
               #and(list[17+(18*p)][0] - list[11+(18*p)][0])/w2 >= -0.25)):#maybe extreme slumping forward

              return "Slumping forward"


           elif (((list[17+(18*p)][0] - list[11+(18*p)][0])/w2 > 0.055)) :

              return "Slouching"
           elif ((  list[17 + (18 * p)][0] - list[5 + (18 * p)][0]) / w2 <-0.023 or((list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 < -0.17)):
              return "head forward"
           elif ((abs(list[17+(18*p)][0] - list[5+(18*p)][0]) /w2 <= 0.15) and  (abs(list[17+(18*p)][0] - list[11+(18*p)][0]) /w2<= 0.045)) :

              return "Right Pose"
           else :

              return "normal head alignment"

         else :
          print("right")
          if (  list[8 + (18 * p)][0] == -1 ):
              if (( list[16 + (18 * p)][0] - list[2+ (18 * p)][0]) / w2 > 0.025 or  ((list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 >0.17 )):
                return "R head forward"
              else :
                  return "R normal head alignmentR"

          if (((list[16+(18*p)][0] - list[8+(18*p)][0])/w2 < -0.05 and (list[16+(18*p)][0] - list[8+(18*p)][0])/w2 >= -0.2)):

            return "R SlouchingR";
          elif ((list[16 + (18 * p)][0] - list[8 + (18 * p)][0]) / w2 > 0.05):
            # and (list[16+(18*p)][0] - list[8+(18*p)][0])/w2 < 0.4)) :
            return "R Slumping forwardR"
          elif ((list[16 + (18 * p)][0] - list[2 + (18 * p)][0]) / w2 > 0.025 or  ((list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 >0.17)):
              return "R head forward"
          elif ((abs(list[16+(18*p)][0] - list[2+(18*p)][0]) /w2 <= 0.1) and(abs(list[16+(18*p)][0] - list[8+(18*p)][0])/w2 <= 0.05)) :

            return "Right PoseR";





          else:
            return "UNDEFINED"

      except Exception as e:
        print(e)


def detectStanding(nn_manager, list, p):
    # print(list)
    score = list
    # print("len(list)")
    # print(len(list))
    leftscore = 10
    rightscore = 10
    w, h = nn_manager.input_size
    w2 = float(w)
    try:
        for i in range(len(left)):
            if (list[left[i] + (18 * p)][2] == -1):
                leftscore = leftscore - 1
            if (list[right[i] + (18 * p)][2] == -1):
                rightscore = rightscore - 1

        if (leftscore > rightscore):
            print("left")

            if ((list[11 + (18 * p)][0] == -1)):
                if ((list[17 + (18 * p)][0] - list[5 + (18 * p)][0]) / w2 < -0.028 or (
                        list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 < -0.17):
                    return "head forward"
                else:
                    return "normal head alignment"

            elif (((list[17 + (18 * p)][0] - list[11 + (18 * p)][0]) / w2 < -0.055)):
                # and(list[17+(18*p)][0] - list[11+(18*p)][0])/w2 >= -0.25)):#maybe extreme slumping forward

                return "Slumping forward"


            elif (((list[17 + (18 * p)][0] - list[11 + (18 * p)][0]) / w2 > 0.055)):

                return "Slumping backward"
            elif ((list[17 + (18 * p)][0] - list[5 + (18 * p)][0]) / w2 < -0.030 or (
                    (list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 < -0.17)):
                return "head forward"
            elif ((abs(list[17 + (18 * p)][0] - list[5 + (18 * p)][0]) / w2 <= 0.15) and (
                    abs(list[17 + (18 * p)][0] - list[11 + (18 * p)][0]) / w2 <= 0.045)):

                return "Right Pose"
            else:

                return "normal head alignment"

        else:
            print("right")
            if (list[8 + (18 * p)][0] == -1):
                if ((list[16 + (18 * p)][0] - list[2 + (18 * p)][0]) / w2 > 0.025 or (
                        (list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 > 0.17)):
                    return "head forwardR"
                else:
                    return "normal head alignmentR"

            if (((list[16 + (18 * p)][0] - list[8 + (18 * p)][0]) / w2 < -0.04 and (
                    list[16 + (18 * p)][0] - list[8 + (18 * p)][0]) / w2 >= -0.2)):

                return "slumping backward";
            elif ((list[16 + (18 * p)][0] - list[8 + (18 * p)][0]) / w2 > 0.028): #0.05
                # and (list[16+(18*p)][0] - list[8+(18*p)][0])/w2 < 0.4)) :
                return "Slumping forwardR"
            elif ((list[16 + (18 * p)][0] - list[2 + (18 * p)][0]) / w2 > 0.03 or (
                    (list[0 + (18 * p)][0] - list[1 + (18 * p)][0]) / w2 > 0.17)):
                return "head forwardR"
            elif ((abs(list[16 + (18 * p)][0] - list[2 + (18 * p)][0]) / w2 <= 0.1) and (
                    abs(list[16 + (18 * p)][0] - list[8 + (18 * p)][0]) / w2 <= 0.05)):

                return "Right Pose";





            else:
                return "UNDEFINED"

    except Exception as e:
        print(e)

def data(command):
    global com
    com=command
global flag

def error(flag):
        if (flag == 1):
            engine.say("Sorry, I did not get that say it again")
            engine.runAndWait()
            engine.stop()
            listen()







def listen():
    with sr.Microphone() as source:
        r.pause_threshold = 1
        print("Talk")
        audio_text = r.listen(source)
        print("Time over, thanks")

        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling

        try:
            command = r.recognize_google(audio_text)

            data(command)
            # using google speech recognition
            print("Text: " + r.recognize_google(audio_text))
        except:

            print("Sorry, I did not get that say it again")
            flag=1
            error(flag)



listen()


global com


def detectScoliosis(nn_manager, list, p):
    # print(list)
    score = list
    # print("len(list)")
    # print(len(list))
    w, h = nn_manager.input_size
    h2 = float(h)
    try:
        left = list[17 + (18 * p)][2] + list[11 + (18 * p)][2] + list[5 + (18 * p)][2]
        right = list[16 + (18 * p)][2] + list[8 + (18 * p)][2] + list[2 + (18 * p)][2]

        if (((abs(list[5 + (18 * p)][1] - list[2 + (18 * p)][1]) / h2 )>=0.018 )
        or (abs(list[8 + (18 * p)][1] - list[11 + (18 * p)][1]) / h2 )>=0.019 )or(abs(list[8 + (18 * p)][1] - list[11 + (18 * p)][1]) / h2 >=0.019):
            return("Scoliosis")


        else:
            return("you dont have Scoliosis ")
    except Exception as e:
        print(e)





def detectBowlegs(nn_manager, list, p):
    # print(list)
    score = list
    # print("len(list)")
    # print(len(list))
    w, h = nn_manager.input_size
    w2 = float(w)
    try:
        if ( (abs(list[9 + (18 * p)][0] - list[12 + (18 * p)][0])/w2 >0.08 ) ):#ankles
            return("Bowlegs")
        else:
            return("you dont have  Bowlegs ")
    except Exception as e:
        print(e)

def detectKyphosis(nn_manager, list, p):
    # print(list)
    score = list
    # print("len(list)")
    # print(len(list))
    w, h = nn_manager.input_size
    w2 = float(w)
    try:
        if ( (abs(list[5 + (18 * p)][0] - list[11 + (18 * p)][0])/w2 >0.025 ) or ((abs(list[2 + (18 * p)][0] - list[11 + (18 * p)][0])/w2 >0.04 ) )):#ankles
            return("kyphosis")
        else:
            return(" yo do not have  Kyphosis ")
    except Exception as e:
        print(e)




def getKeypoints(probMap, threshold=0.2):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []
    contours = None
    try:
        # OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    #print("key point********************************")
    #print(keypoints)
    return keypoints


def getValidPairs(outputs, w, h, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.2
    conf_th = 0.4

    for k in range(len(mapIdx)):

        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))
        candA = detected_keypoints[POSE_PAIRS[k][0]]

        candB = detected_keypoints[POSE_PAIRS[k][1]]

        nA = len(candA)
        nB = len(candB)

        if (nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][
                        2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


threshold = 0.3
nPoints = 18
detected_keypoints = []


def decode(nn_manager, packet):
    outputs = to_tensor_result(packet)["Openpose/concat_stage7"].astype('float32')
    w, h = nn_manager.input_size
    #print("output")

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    #print("nPoints")
    #print((nPoints))
    kx = []
    ky = []
    kscore = []


    for part in range(nPoints):
        probMap = outputs[0, part, :, :]

        probMap = cv2.resize(probMap, (w, h))  # (456, 256)
        #print("!!!!!")
        #print(probMap)
        keypoints = getKeypoints(probMap, threshold)

        #print("len(keypoints")
        #print((keypoints))
        #print((keypoints)) #try 2 persons ,, and empty places
        #print( keypointsMapping[part])

        keypoints_with_id = []


        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)
    #print("what is detected point")
    #print(" keypoints??????????")
    #print((detected_keypoints))
    for i in range(len(detected_keypoints)):
        try:
         if(len(detected_keypoints[i])<1):
             kx.append(500)
             ky.append(500)
             kscore.append(0)
         else:

            #print((detected_keypoints[i]))
            #print((detected_keypoints[i][0][0]))
            kscore.append(detected_keypoints[i][0][2])
            kx.append(detected_keypoints[i][0][0])
            ky.append(detected_keypoints[i][0][1])

        except:
          print("exception")

    valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, detected_keypoints)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)
    keypoints_limbs = [detected_keypoints, personwiseKeypoints, keypoints_list]
    #print("len(kx)")
    #print(len(kx))
    return keypoints_limbs



def draw(nn_manager, keypoints_limbs, frames):

    for name, frame in frames:
        if name == nn_manager.source and len(keypoints_limbs) == 3:
            detected_keypoints = keypoints_limbs[0]

            personwiseKeypoints = keypoints_limbs[1] #count
            #print("personwiseKeypoints")
            #print(personwiseKeypoints)
            keypoints_list = keypoints_limbs[2] #real point
            #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            #try:
                #print(len(detected_keypoints))
                #print(detected_keypoints)

            #except:
                #print(":(")
            #print("what is detected key point")
            #print(len(detected_keypoints))
            #print((detected_keypoints))
            for i in range(nPoints):
                #print("len(detected_keypoints[i])")  # of people
                #print(len(detected_keypoints[i]))
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (100, 150)
                    fontScale = 0.5
                    fontColor = (255, 255, 255)
                    lineType = 1
                    print(keypointsMapping[i])
                    print(str(detected_keypoints[i][j][0])+", "+str(detected_keypoints[i][j][1]))
                    cv2.putText(frame, str(detected_keypoints[i][j][0])+", "+str(detected_keypoints[i][j][1]),
                                detected_keypoints[i][j][0:2],
                                font,
                                fontScale,
                                fontColor,
                                lineType)
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    #print(personwiseKeypoints[n][i])
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

            finalPeoplePoint=[]
            for n in range(len(personwiseKeypoints)):

                for i in range(18) :
                    if (personwiseKeypoints[n][i] ==-1) :
                        finalPeoplePoint.append([-1,-1,-1,i])
                    else :
                      if ((len(detected_keypoints[i])!=0 )and len(detected_keypoints[i])>n and personwiseKeypoints[n][i]==detected_keypoints[i][n][3]) :
                          finalPeoplePoint.append([detected_keypoints[i][n][0],detected_keypoints[i][n][1],detected_keypoints[i][n][2],i])
                      elif (len(detected_keypoints[i])==0):
                          finalPeoplePoint.append([-1, -1, -1, i])
            #print("finalPeoplePoint")
            #print(finalPeoplePoint)
            leng = int(len(finalPeoplePoint)/18)

            for p in range(leng):

                 #print(finalPeoplePoint[8 + (18*p)][2])
                 #print("p")
                 #print(p)
                 print("Status")
                 if  ( com=="scoliosis"):

                  writeText(frame,detectScoliosis(nn_manager,finalPeoplePoint,p))
                  engine.say(detectScoliosis(nn_manager,finalPeoplePoint,p))
                  engine.runAndWait()
                 elif (com=="kyphosis"):

                     writeText(frame, detectKyphosis(nn_manager, finalPeoplePoint, p))
                     engine.say(detectKyphosis(nn_manager, finalPeoplePoint, p))
                     engine.runAndWait()

                 elif(com=="bowlegs"):
                     writeText(frame, detectBowlegs(nn_manager, finalPeoplePoint, p))
                     engine.say(detectBowlegs(nn_manager, finalPeoplePoint, p))
                     engine.runAndWait()
                 else :
                     status = "UNDEFINED"
                     for p in range(leng):
                         # print(finalPeoplePoint[8 + (18*p)][2])
                         print("p")
                         print(p)
                         print("Status")
                         sum = 0;
                         for i in range(18):
                             sum = sum + finalPeoplePoint[i + (18 * p)][2]
                         print("suuuuuuuuuuuuum")
                         s = IsSettingOrStanding(finalPeoplePoint, p)
                         if (s == "sitting"):
                             status = detectSitting(nn_manager, finalPeoplePoint, p)
                         if (s == "standing"):
                             status = detectStanding(nn_manager, finalPeoplePoint, p)

                         p1 = finalPeoplePoint[0 + (18 * p)][0]
                         p2 = finalPeoplePoint[0 + (18 * p)][1]
                         if (p2 < 50 and p2 > 0):
                             p2 = 330
                         else:
                             p2 = 20
                         if (p1 < 0):
                             p1 = finalPeoplePoint[2 + (18 * p)][0] + 50
                         txtContent.append(status)
                         txtPlace.append((abs(p1 - 20), p2))

                     for i in range(nPoints):
                         # print("len(detected_keypoints[i])")  # of people
                         # print(len(detected_keypoints[i]))
                         for j in range(len(detected_keypoints[i])):

                             if i == 0 or i == 2 or i == 5 or i == 8 or i == 11 or i == 16 or i == 17:
                                 if (status == "UNDEFINED"):
                                     cv2.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                                 if (status == "Right Pose" or status == "normal head alignment"):
                                     colors[i] = green
                                     cv2.circle(frame, detected_keypoints[i][j][0:2], 7, green, -1, cv2.LINE_AA)

                                 else:

                                     cv2.circle(frame, detected_keypoints[i][j][0:2], 7, red, -1, cv2.LINE_AA)

                                     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                             else:
                                 cv2.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                             font = cv2.FONT_HERSHEY_SIMPLEX
                             bottomLeftCornerOfText = (100, 150)
                             fontScale = 0.5
                             fontColor = (255, 255, 255)
                             lineType = 1
                             print(keypointsMapping[i])
                             print(str(detected_keypoints[i][j][0]) + ", " + str(detected_keypoints[i][j][1]))
                     for r in range(len(txtPlace)):
                         print("txt content len")
                         print(len(txtPlace))
                         if (txtContent[r] == "UNDEFINED"):
                             writeText(frame, status, black, txtPlace[r])
                         if (txtContent[r] == "Right Pose" or status == "normal head alignment"):
                             writeText(frame, status, green, txtPlace[r])
                         else:
                             writeText(frame, status, red, txtPlace[r])
                     for i in range(17):
                         for n in range(len(personwiseKeypoints)):
                             index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                             # print(personwiseKeypoints[n][i])
                             if -1 in index:
                                 continue
                             B = np.int32(keypoints_list[index.astype(int), 0])
                             A = np.int32(keypoints_list[index.astype(int), 1])
                             cv2.line(frame, (B[0], A[0]), (B[1], A[1]), red, 3, cv2.LINE_AA)
                             if i == 6 or i == 9 or i == 12 or i == 13 or i == 15 or i == 8 or i == 11:
                                 if (status == "UNDEFINED"):
                                     cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

                                 if (status == "Right Pose" or status == "normal head alignment"):

                                     cv2.line(frame, (B[0], A[0]), (B[1], A[1]), green, 4, cv2.LINE_AA)

                                 else:
                                     cv2.line(frame, (B[0], A[0]), (B[1], A[1]), red, 4, cv2.LINE_AA)

                             else:
                                 cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)









