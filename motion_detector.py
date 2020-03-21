import cv2

first_frame = None

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    # frame color edits
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # set first frame to gray
    if first_frame is None:
        first_frame = gray
        continue
    
    # comparing difference
    delta_frame = cv2.absdiff(first_frame, gray)

    cv2.imshow("Capturing", delta_frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows
