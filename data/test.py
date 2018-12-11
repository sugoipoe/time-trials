import numpy as np
import cv2 as cv
for hundreds in range(3):
    for tens in range(10): 
        #cap = cv.VideoCapture('fortnite_single_90.mp4')
        #cap = cv.VideoCapture('sliced_fortnite_90_mp4s/frames003x.mp4')
        cap = cv.VideoCapture('sliced_fortnite_90_mp4s/frames0{}{}x.mp4'.format(hundreds, tens))
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
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
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        for i in range(10):
            ret,frame = cap.read()

            # frame is a multi-dimensional array, until .read() finishes and it's "None".
            # Unfortunately, python will not let me compare the array to "None", i.e. frame == None, so we do this
            # as a workaround. When frame is None, None.any() will throw an AttributeError.
            try:
                frame.any()
            except AttributeError:
                print("No frame, aborting!")
                #import pdb; pdb.set_trace()
                break
            #import pdb; pdb.set_trace()
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
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv.add(frame,mask)
            #import pdb; pdb.set_trace()
            cv.imshow('frame',img) # this line
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        cv.destroyAllWindows()
        cap.release()
