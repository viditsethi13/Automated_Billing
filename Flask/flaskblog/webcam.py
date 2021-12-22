import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.imwrite('messigray.png',frame)
        print("Image Saved")
        
    c = cv2.waitKey(1)
    if c == 27:
        break
    elif c == 32:
        x=1;

cap.release()
cv2.destroyAllWindows()