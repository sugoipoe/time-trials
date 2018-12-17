import numpy as np
import cv2 as cv

import argparse
import math

import pdb

FRAMES_PER_CLIP=10
DATA_DIR='/home/sugoipoe/Projects/time-trials/data/sliced_fortnite_90_mp4s/'
FILENAME_FORMAT='frames0{}{}x.mp4'

def show_corners(corners, frame):
# corners is a numpy array with coordinates for each selected corner
    for point in corners:
        a, b = point[0].ravel()
        frame = cv.circle(frame, (a, b), 3, color[0].tolist(),-1)
        img = cv.add(frame, mask) # Add this back if you want to see the corners

        
    cv.imshow('frame', img) # Add this back for corners
    cv.waitKey(1) # Add this back for corners
    raw_input("Press something to continue") # Wait for user to continue

    return

def velocity_between_points(x1, y1, x2, y2):
# This function returns a tuple, (distance, direction), where direction is the degrees away from vertical
# i.e., if point A is directly below point B (A is original), then direction = 0 (0 degrees for up, 180 for down, etc.)

    distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    horiz_distance = x2 - x1
    vert_distance = y2 - y1
    direction = math.atan2(vert_distance, horiz_distance)

    print(distance, direction)
    return distance, direction


def optical_flow(args):
    # Loop through video files
    for hundreds in range(3):
        for tens in range(10): 
            cap = cv.VideoCapture(DATA_DIR + FILENAME_FORMAT.format(hundreds, tens))
            print(DATA_DIR + FILENAME_FORMAT.format(hundreds, tens))


            # Set default params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 4500,
                                   qualityLevel = 0.0005,
                                   minDistance = 5,
                                   blockSize = 7 )

            # Override default params with args
            if args.corners:
                feature_params['maxCorners'] = args.corners
            if args.quality:
                feature_params['qualityLevel'] = args.quality
            if args.mindistance:
                feature_params['minDistance'] = args.mindistance
            

            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

            # Create some random colors
            color = np.random.randint(0,255,(100,3))

            # Take first frame and find corners in it
            ret, old_frame = cap.read()
            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            print("Selected {} corners.".format(len(p0)))

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            for i in range(FRAMES_PER_CLIP):
                ret,frame = cap.read()

                if args.show_corners:
                    show_corners(p0, frame)


                # frame is a multi-dimensional array, until .read() finishes and it's "None".
                # Unfortunately, python will not let me compare the array to "None", i.e. frame == None, so we do this
                # as a workaround. When frame is None, None.any() will throw an AttributeError.
                try:
                    frame.any()
                except AttributeError:
                    print("No frame, skipping to next video")
                    break
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # calculate optical flow
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    velocity_between_points(a,b,c,d)
                    #mask = cv.line(mask, (a,b),(c,d), color[i % len(color)].tolist(), 2) # Add this for random colors
                    #frame = cv.circle(frame,(a,b),3,color[i % len(color)].tolist(),-1) # Add this for random colors
                    mask = cv.line(mask, (a,b),(c,d), color[0].tolist(), 2) # Add this for random colors
                    frame = cv.circle(frame,(a,b),3,color[0].tolist(),-1) # Add this for random colors
                img = cv.add(frame,mask)
                #import pdb; pdb.set_trace()
                cv.imshow('frame',img) # this line draws the illustration
                k = cv.waitKey(10) & 0xff
            #raw_input("Press something to continue") # Wait for user to continue
                if k == 27:
                    break
                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)

def main():
    parser = argparse.ArgumentParser(description='Do optical flow on fortnite videos')
    parser.add_argument('--corners', dest='corners', type=int)
    parser.add_argument('--quality', dest='quality', type=float)
    parser.add_argument('--mindistance', dest='mindistance', type=int)

    parser.add_argument('--show_corners', dest='show_corners', action='store_true', help='Show Shi Tomasi corners')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Show debug messages')

    args = parser.parse_args()

    optical_flow(args)


if __name__ == '__main__':
    main()
