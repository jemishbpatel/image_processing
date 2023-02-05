import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import bisect

NEW_LOW = 0.2
NEW_HIGH = 0.8
cap = cv2.VideoCapture('/home/jemish/Downloads/research/clips/first_experiment/near_color/video2.mp4')

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('right.mp4', 
                         cv2.VideoWriter_fourcc('M','J','P','G'),
                         20, size)

while (cap.isOpened()):
    # Get a video frame
    hasFrame, frame = cap.read()


    if hasFrame == True:
	b,g,r = cv2.split( frame )
	frameblue = cv2.normalize( b, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	arrblue = np.asarray(frameblue )
	arr2blue = imadjust(arrblue, arrblue.min(),arrblue.max() , NEW_LOW, NEW_HIGH )

	framegreen = cv2.normalize( g, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	arrgreen = np.asarray(framegreen )
	arr2green = imadjust(arrgreen, arrgreen.min(),arrgreen.max() , NEW_LOW, NEW_HIGH )

	framered = cv2.normalize( r, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	arrred = np.asarray(framered )
	arr2red = imadjust(arrred, arrred.min(),arrred.max() , NEW_LOW, NEW_HIGH )

	merged = cv2.merge([arr2blue, arr2green, arr2red])
	
	mergedCopy = (merged * 255).astype(np.uint8)
	result.write(mergedCopy)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break	

    else:
        break

cap.release()
result.release()
cv2.destroyAllWindows()
