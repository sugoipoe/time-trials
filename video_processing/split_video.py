# This utility takes a video, and splits it into multiple files
import ffmpeg

FILENAME='fortnite_single_90.mp4'
FRAMES_PER_SPLIT=10

def split_video(filename, frames_per_split):
  outfile = "thumbb0001"
  stream = ffmpeg.input(filename).output(outfile).run()


split_video(FILENAME, FRAMES_PER_SPLIT)
