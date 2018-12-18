import numpy as np
import cv2 as cv

import argparse
import math
import os
import logging

from collections import defaultdict

import pdb

# TODO list
# 1. Take full clips and separate them here into configurable #frames


FRAMES_PER_CLIP=10
NUM_CLIPS = 30
PROJECT_HOME='{homedir}/Projects/time-trials/'.format(homedir=os.getenv("HOME"))
DATA_DIR='{project_dir}/data/sliced_fortnite_90_mp4s/'.format(project_dir=PROJECT_HOME)
FILENAME_FORMAT='frames0{:02d}x.mp4' # frame000x/frame001x...frame031x...

def bucket_corners(velocity_vector_list):
# `bucket_corners` takes a list of (distance, direction) tuples and roughly buckets them
    return


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
# i.e., if point A is directly below point B (A is original), then direction = 90. -> == 0, <- == 180/-180, down == -90

    distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    horiz_distance = x2 - x1
    vert_distance = y2 - y1
    direction = math.degrees(math.atan2(vert_distance, horiz_distance))

    print(distance, direction)
    return distance, direction

def process_clip(filename, feature_params, lk_params, color):
  # Buckets for filtering out corners in the foreground
  filter_bucket = defaultdict(list)

  cap = cv.VideoCapture(filename)
  
  # Take first frame and find corners in it
  ret, old_frame = cap.read()
  old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
  p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  logging.debug("Selected {} corners".format(len(p0)))

  # Create a mask image for drawing purposes
  mask = np.zeros_like(old_frame)

  for i in range(FRAMES_PER_CLIP):
      ret,frame = cap.read()

      # Display corners w/o movement lines at the start
      #if args.show_corners:
      #    show_corners(p0, frame)

      # frame is converted to None when .read() completes (no frames left)
      try:
          frame.any()
      except AttributeError:
          print("No frame, skipping to next video")
          break

      # optical flow requires a grayscale image -- convert here
      frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      # calculate optical flow
      p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]
      
      # draw the tracks
      for i,(new,old) in enumerate(zip(good_new,good_old)):
          # Unpack x/y coordinates for corners before and after
          a,b = new.ravel()
          c,d = old.ravel()

          # Calculate the velocity vector to filter corners into buckets
          distance, direction = velocity_between_points(a,b,c,d)
          filter_bucket[int(direction)].append((distance, direction))

          # Generate lines (no gameplay image) between old and new corners
          mask = cv.line(mask, (a,b),(c,d), color[0].tolist(), 2)
          # Overlay corner locations on the original frame
          frame = cv.circle(frame,(a,b),3,color[0].tolist(),-1)

      # Overlay lines on the frame w/ corner circles
      img = cv.add(frame, mask)
      cv.imshow('frame', img)
      k = cv.waitKey(10) & 0xff
      if k == 27:
          break
      # Keep old frame to compare against the next
      old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1,1,2)


def optical_flow(args):
    # Set default params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 4500,
                           qualityLevel = 0.0005,
                           minDistance = 5,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Override default params with args
    if args.corners:
        feature_params['maxCorners'] = args.corners
    if args.quality:
        feature_params['qualityLevel'] = args.quality
    if args.mindistance:
        feature_params['minDistance'] = args.mindistance

    # Process each clip
    for clip_id in range(NUM_CLIPS):
        filename = DATA_DIR + FILENAME_FORMAT.format(clip_id)
        print(filename)
        process_clip(filename, feature_params, lk_params, color)



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
