import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')


while(cap.isOpened()):

    ret, frame = cap.read()
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blur, 50, 150)
    
    
    mask = np.zeros_like(edges)
    height, width = edges.shape
    region_of_interest = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    vertices = np.array([region_of_interest], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 20
    max_line_gap = 20
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    
    cv2.imshow('Lane Detection', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
