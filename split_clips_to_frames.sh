#this script creates sequences of frames from the videos in clips directory
#using FFMPEG
#If you don't have ffmpeg installed on your system
#run sudo apt-get install ffmpeg

CLIPS_DIR=JAAD_clips  #path to the directory with mp4 videos
FRAMES_DIR=images  #path to the directory for frames

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
ffmpeg -i $file -start_number 0 -f image2 -qscale 1 ${FRAMES_DIR}/$fname/%05d.png

done