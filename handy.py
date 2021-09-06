import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pynput.keyboard import Controller, Key
import time


model = load_model('finger.hdf5')

cap = cv2.VideoCapture(0)

keyboardCont = Controller()
checkFlag = True
start = time.time()

while True:

    # This will be the frames that are taken from Live feed
    isTrue, frame = cap.read()
    frame = cv2.flip(frame, 1)
    show = frame[50:200, 400:550]

    frame = cv2.blur(frame, (2, 2))
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayScale = grayScale[50:200, 400:550]

    isTruet, mask = cv2.threshold(grayScale, 120, 255, cv2.THRESH_BINARY_INV)
    mask = mask / 255.0
    mask = cv2.resize(mask, (128, 128))
    mask = mask.reshape(-1, 128, 128, 1)

    outputPrediction = np.zeros((200, 400, 3))

    #Predicting the Count of fingers according to the model trained
    result = model.predict(mask)
    res = np.argmax(result)

    cv2.putText(outputPrediction, "Result: {}".format(res), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    if checkFlag:
        if res == 0:
            keyboardCont.press(Key.space)
            keyboardCont.release(Key.space)
            checkFlag = False
        elif res == 1:
            keyboardCont.press(Key.up)
            keyboardCont.release(Key.up)
            checkFlag = False
        elif res == 2:
            keyboardCont.press(Key.down)
            keyboardCont.release(Key.down)
            checkFlag = False
        elif res == 3:
            keyboardCont.press(Key.left)
            keyboardCont.release(Key.left)
            checkFlag = False
        elif res == 4:
            keyboardCont.press(Key.right)
            keyboardCont.release(Key.right)
            checkFlag = False
    ############################

    cv2.imshow("main", show)
    cv2.imshow("result", mask.reshape(128,128))
    cv2.imshow("", outputPrediction)

    end = time.time()

    #To give keyboard input in the gap of 2 seconds
    if (end - start) > 2:
        start = end
        checkFlag = True

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()