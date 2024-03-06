# Gaze Data and Focus on Faces
Analyzing gaze data to determine the percentage of gaze locations on faces in each frame
## Loading Data
Save the video file and gaze data file to the same directory as face_detection.py. The gaze data file must have the headers:

```frame_num shot_num x y timestamp subject eyetracker_valid in_frame subject_valid_for_clip film```

To extract the frames from the video file, you can run

``` ffmpeg -i myclip.mp4  'path/to/where/i/want/frames/myclip_%d.jpg' ```

Ensure that all of the file names only contain a single number that corresponds to the frame number in the gaze data. Do not include any other numbers in the name. 

## Installation
```pip install -r requirements.txt```

## Analyzing the Data
Run face_detector.py

### Analyzing the Entire Dataset
To get the percentage of gaze locations for each frame, run 

```analyze_directory(frames_directory, data_file_path, film_name)```

This will save a csv file to your directory.

### Visualizing Bounding Boxes
To view where the bounding boxes on a frame are, run

```detect_faces(frame_path, visualize=True)```


