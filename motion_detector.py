import cv2
import pandas
from datetime import datetime

first_frame = None
status_list = [None, None]
presence_list = []
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0

    # frame color edits
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # set first frame to gray
    if first_frame is None:
        first_frame = blurred_gray
        continue
    

    delta_frame = cv2.absdiff(first_frame, blurred_gray)                        # comparing difference
    tresh_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]     # adding treshold effect
    tresh_frame = cv2.dilate(tresh_frame, None, iterations=2)                   # clearing frame

    # touple of contours
    (cnts,_) = cv2.findContours(tresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        presence_list.append(datetime.now())
    
    if status_list[-1] == 0 and status_list[-2] == 1:
        presence_list.append(datetime.now())


    cv2.imshow("Gray Frame", blurred_gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Treshold Frame", tresh_frame)
    cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            presence_list.append(datetime.now())
        break
    
print(presence_list)

for i in range(0, len(presence_list), 2):
    df = df.append({"Start":presence_list[i], "End":presence_list[i+1]}, ignore_index = True)

df.to_csv("presence.csv")

video.release()
cv2.destroyAllWindows
