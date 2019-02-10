import cv2
import numpy as np

#=============== Variable Mouse ==================#
drawing = False
point1 = ()
point2 = ()

drawingTwo = False
pointTwo_1 = ()
pointTwo_2 = ()
Mouse_count = False
#================================================#
def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing
    global pointTwo_1, pointTwo_2, drawingTwo, Mouse_count

    #----------Mouse 1-------
    if Mouse_count == False:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing is False:
                drawing = True
                point1 = (x, y)
            #else:
                #drawing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                point2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            Mouse_count = True
            
    #----------Mouse 2-------#
    if Mouse_count == True:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawingTwo is False:
                drawingTwo = True
                pointTwo_1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawingTwo is True:
                pointTwo_2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if drawingTwo is True:
                drawingTwo = False
                Mouse_count = False
            
            
#================================================#

#create VideoCapture object and read from video file
cap = cv2.VideoCapture('cars.mp4')

cv2.namedWindow("Detecion Car")
cv2.setMouseCallback("Detecion Car", mouse_drawing)

while True:
    ret, frame = cap.read()
    car_cascade = cv2.CascadeClassifier('cars.xml')

    #============================== ROI One ============================#
    if point1 and point2:

        #Rectangle marker
        r = cv2.rectangle(frame, point1, point2, (100, 50, 200), 5)
        frame_ROI = frame[point1[1]:point2[1],point1[0]:point2[0]]

        #------------------Detect car ROI-------------------#
        if drawing is False:
            #convert video into gray scale of each frames
            ROI_grayscale = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
            #detect cars in the video
            cars_ROI = car_cascade.detectMultiScale(ROI_grayscale, 1.1, 3)
            for (x,y,w,h) in cars_ROI:
                cv2.rectangle(frame_ROI,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame_ROI, "Number of cars: " + str(cars_ROI.shape[0]), (10,frame_ROI.shape[0] -25), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(0,255,0), 1)
       #-------------------------------------------------#
    #==================================================================#

    #============================== ROI Two ============================#
    if pointTwo_1 and pointTwo_2:
        #Rectangle marker
        r2 = cv2.rectangle(frame, pointTwo_1, pointTwo_2, (0, 255, 255), 5)
        frameTWO_ROI = frame[pointTwo_1[1]:pointTwo_2[1],pointTwo_1[0]:pointTwo_2[0]]
        #---------------------Detect car ROI2----------------------------#
        if drawingTwo is False:
            #convert video into gray scale of each frames
            frame_grayscale = cv2.cvtColor(frameTWO_ROI, cv2.COLOR_BGR2GRAY)
            #detect cars in the video
            carsTwo_ROI = car_cascade.detectMultiScale(frame_grayscale, 1.1, 3)
            #to draw arectangle in each cars
            for (x,y,w,h) in carsTwo_ROI:
                cv2.rectangle(frameTWO_ROI,(x, y),(x+w,y+h),(255,255,100),2)
                cv2.putText(frameTWO_ROI, "Number of cars: " + str(carsTwo_ROI.shape[0]), (10,frameTWO_ROI.shape[0] -25), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,100), 1)
        #------------------------------------------------------------#
    #==================================================================#
    cv2.imshow("Detecion Car", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
