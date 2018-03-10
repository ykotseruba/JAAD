#this script creates sequences of frames from the videos in clips directory
#using FFMPEG

############### CHANGE THESE PATHS #############################

CLIPS_DIR=path_to_JAAD_clips  #full path to the directory with mp4 videos
FRAMES_DIR=path_to_JAAD_frames  #full path to the directory for frames

################################################################


for file in ${CLIPS_DIR}/*.mp4
do
if [ -d ${file} ]; then
continue;
fi

#make a directory to save frame sequences
mkdir ${FRAMES_DIR}

filename=$(basename "$file")
fname="${filename%.*}"
echo $fname

#create a directory for each frame sequence
mkdir ${FRAMES_DIR}/$fname
ffmpeg -i $file -f image2 -qscale 1 ${FRAMES_DIR}/$fname/frame_%04d.png

done
