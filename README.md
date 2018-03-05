# JAAD Dataset Annotations
This repository contains annotation data for JAAD (Joint Attention in Autonomous Driving) dataset.  

JAAD dataset aims to provide samples for pedestrian detection, pedestrian action and gesture recognition, and behavioral studies of traffic participants. The data consists of 346 high-resolution video clips (5-15s) with annotations showing various situations typical
for urban driving. The videos are recorded using dashboard mounted cameras.

All video sequences can be downloaded in mp4 and seq format from our project site (http://data.nvision2.eecs.yorku.ca/JAAD_dataset/).
To convert the video clips to image sequences, use the script 'split_clips_to_frames.sh'.

## Bounding boxes
The bounding boxes are provided for all pedestrians in the video (and very few vehicles) in vbb format which require a Piotr Dollar's Computer Vision Matlab Toolbox available at (https://pdollar.github.io/toolbox/).  

There are two folders with bounding boxes.
vbb_part contains bounding boxes with occlusion flags set to 1 for partial or full occlusion (>75% of the object occluded).  
vbb_full contains bounding boxes ONLY with full occlusion.  

To open vbb files in Matlab use the following command:
```
A = vbb('vbbLoad', 'vbb_part/video_0001.vbb');
```

There are three types of labels in the vbb files:  
car, car1, car2, ... - vehicles (only few vehicles that interact directly with the driver are labeled)  
pedestrian, pedestrian1, pedestrian2, ... - pedestriansl with behavioral tags  
ped1, ped2, .... - other pedestrians visible in the scene (bystanders) without behavioral tags and attributes  

To view vbb boxes with .seq files use the vbbLabeler utility provided with the toolbox.  

## Traffic scene elements

We provide traffic_scene_elements.txt file  which lists scene elements for each video with corresponding frame numbers.  
The text is formatted as follows:  
video_id, attr_id: start_frame-end_frame; attr_id: start_frame-end_frame;  
Note: if no range is provided, the scene element is visible in all frames of the video.  

e.g. video_0005, 1: 1-30; 1: 90-240; 7;
stop sign (1) is visible in frames 1-30 and 90-240
the whole scene is filmed in a parking lot (7)

## Pedestrian attributes
For all pedestrians that have associated behavioral data we provide additional attributes in a text file (pedestrian_attributes.txt).  
Each line lists attributes (comma-separated) for a single pedestrian in the following order:  
video_id, pedestrian_id, group_size, direction, designated, signalized, gender, age, num_lanes, traffic direction, intersection, crossing

* video_id, pedestrian_id, gender (male/female and n/a for small children) and age (child/young/adult/senior) are self-explanatory
* group_size: size of the group that the pedestrian is part of (moving or standing together)
* direction: indicates whether the pedestrian is moving along the direction of car's movement (LONG), crossing in front of the car (LAT) or standing (n/a)
* designated: the location where the pedestrian is moving/standing is designated for crossing (D) or non-designated (ND)
* signalized: the location where the pedestrian is moving/standing is signalized (S), i.e. has a stop sign or traffic lights, or not signalized (NS)
* num_lanes: number of lanes at the place where the pedestrian is moving/standing
* traffic direction: OW - one way, TW - two way
* intersection: yes - crossing at the intersection and no otherwise
* crossing: 1 - pedestrian completes crossing, 0 - pedestrian does not cross, -1 - no intention of crossing (e.g. waiting at the bus stop, talking to somebody at the curb)

When there is no pedestrians in the video, all attributes are set to "n/a".


## Behavioral annotations
This data is produced using BORIS 2 (http://www.boris.unito.it) - event logging software for video observations. We provide both the BORIS files in the original tsv text format and xml...
Each file contains the name of the video file (e.g. video_0001.mp4), independent variables and timestamped observations.
The following types of tags and their possible values are defined for each video:

1. location		(indoor/plaza/street)  indoor refers to parking, plaza to outdoor parking (e.g. near mall)

2.	weather  	(cloudy/clear/rain/snow)   n/a is set for indoor

3. time_of_day	 (daytime/nighttime) 	n/a for indoor

4. road_condition (snow/rain/dry)	whether the road surface is covered in snow/water or is dry..

The following behaviors are defined for all subjects: clear path, crossing, handwave, look, looking, moving fast, moving slow, walking, nod, signal, slow down, speed up, standing, stopped. Some actions are capitalized to distinguish actions that happen on the road vs actions on the sidewalk (e.g. STANDING means that the pedestrian is waiting beyond the curb).  

Behavioral data can be downloaded in the original BORIS tsv format and xml.  
Below is an example of behavioral data in xml format.

```
xml
<?xml version="1.0" encoding="utf-8"?>
<video FPS="29.97" filename="video_0001.mp4" id="video_0001" length_sec="20.02" num_frames="600">
   <tags>
      <time_of_day val="daytime"/>
      <weather val="cloudy"/>
      <location val="plaza"/>
      <road_condition val="dry"/>
   </tags>
   <subjects>
      <Driver/>
      <pedestrian1/>
      <pedestrian2/>
   </subjects>
   <actions>
      <Driver>
         <action end_frame="57" end_time="1.9019" id="moving slow" start_frame="1" start_time="0"/>
         <action end_frame="141" end_time="4.7047" id="decelerating" start_frame="58" start_time="1.9353"/>
      </Driver>
      <pedestrian1>
         <action end_frame="364" end_time="12.133" id="standing" start_frame="1" start_time="0.02"/>
         <action end_frame="473" end_time="15.773" id="looking" start_frame="444" start_time="14.8"/>
      </pedestrian1>
      <pedestrian2>
         <action end_frame="70" end_time="2.336" id="walking" start_frame="1" start_time="0.02"/>
      </pedestrian2>
   </actions>
</video>
```

Xml files can be read in MATLAB using xml2struct.m script available at (https://www.mathworks.com/matlabcentral/fileexchange/28518-xml2struct)

### Citing us

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{rasouli2017they,
  title={Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={206--213},
  year={2017}
}

@article{kotseruba2016joint,
  title={Joint attention in autonomous driving (JAAD)},
  author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
  journal={arXiv preprint arXiv:1609.04741},
  year={2016}
}
```

## Authors

* **Amir Rasouli**
* **Yulia Kotseruba**

Please send email to yulia_k@cse.yorku.ca or aras@cse.yorku.ca if there are any problems with downloading or using the data.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
