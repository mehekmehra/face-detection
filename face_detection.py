import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import csv

    
def detect_faces(image_path, visualize):
    """ puts bounding boxes around faces in the specified image.
        inputs
        ------
        image_path: path to the image
        visualize: boolean referring to whether or not you want the image with bounding boxes to be displayed

        outputs
        -------
        boxes: an array of arrays where each subarray corresponds to a bounding box for each face in the image.
               the subarray is formatted as [startX, startY, endX, endY]
        h: the height of the input image
        w: the width of the input image
    """
    image = cv2.imread(image_path)

    classifier = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    classifier.setInput(blob)
    faces = classifier.forward()

    (h, w) = image.shape[:2]
    boxes = []

    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        # confidence chosen to be 0.5 after a few tests, still misses a lot of faces
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes += [[startX, startY, endX, endY]]
            # puts the bounding box on the image if necessary
            if visualize:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # shows the image with the bounding boxes on the faces
    if visualize:
        plt.figure(figsize=(20,10))
        cv2.imshow("Detected Faces", image)
    
    return boxes, h, w
  
def analyze_directory(directory, data_path, film_name):
    """ creates a csv file with the frame of the film and the percentage of gaze locations on a detected face 
        in that frame. If no faces are detected, the percentage is -1.
        inputs
        ------
        directory: the path to the directory in which the frames are stored
        data_path: path to the file containing all of the gaze locations. The file must have the following headers:
                   frame_num shot_num x y timestamp subject eyetracker_valid in_frame subject_valid_for_clip film
        film_name: the name of the film (lowercase)
    """
    files = os.listdir(directory)
    frames = []
    percentages = []
    plt_frames = []
    plt_percentages = []
    film_data = filter_by_film(data_path, film_name)
    for image_name in files:
        image_path = directory + "/" + image_name
        # pulls the number from the name to get the frame number. assumes that there are no other numbers in the name
        frame_num = re.findall(r'\d+', image_name)[0]
        percent = percent_in_box(data_path, frame_num, film_data, image_path)
        percentages += [percent]
        frames += [frame_num]
        if percent != -1:
            plt_frames += [int(frame_num)]
            plt_percentages += [percent]

        # to test with less computation because this isn't super efficient
        # if len(frames) == 50:
        #     break
    
    # writing results to a csv file
    data_rows = zip(frames, percentages)
    csv_path = 'face_detection_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Percent Gaze on Face']) 
        writer.writerows(data_rows) 
    
    return plt_frames, plt_percentages
    
def plot_percentages(directory, data_path, film_name):
    frames, percentages = analyze_directory(directory, data_path, film_name)
    print(len(frames), len(percentages))
    np_frames = np.array(frames)
    np_percentages = np.array(percentages)
    plt.plot(np_frames, np_percentages, 'o')
    plt.xlabel("Frame")
    plt.ylabel("Gaze Locations on a Face (%)")
    plt.show()


def filter_by_film(data_path, film_name):
    """ filters the data file for entries associated with the relevant film. 
        returns an array of the entries for the film where each entry is a subarray
        inputs
        ------
        data_path: path to the file containing all of the gaze locations. The file must have the following headers:
                   frame_num shot_num x y timestamp subject eyetracker_valid in_frame subject_valid_for_clip film
        film_name: the name of the film (lowercase)
        outputs
        -------
        film_data: an array of subarrays where each subarray corresponds to one line in the text file
                   corresponding to the film
    """
    film_col = -1
    film_data = []
    with open(data_path, 'r') as file:
        next(file)
        for line in file:
            cols = line.split()
            # filters for the the movie
            if cols[film_col] == film_name:
                film_data += [cols]
    return film_data

def percent_in_box(data_path, frame_num, film_data, image_path):
    """ returns the percentage of gaze locations within bounding boxes for a specified image
        inputs
        ------
        data_path: path to the file containing all of the gaze locations. The file must have the following headers:
                   frame_num shot_num x y timestamp subject eyetracker_valid in_frame subject_valid_for_clip film
        frame_num: the frame number that the image refers to
        film_data: an array of subarrays where each subarray corresponds to one line in the text file
                   corresponding to the film
        image_path: the path to the image

        outputs
        -------
        percentage_in_box: the percent of gaze locations within boxes for the specified frame. 
                           returns -1 if there are no bounding boxes in an image
    """
    in_box = 0
    total = 0
    frame_col = 1
    
    x_col = 3
    y_col = 4
    boxes, h, w = detect_faces(image_path, False)

    if boxes:
        for cols in film_data:
            # filters for the frame
            if cols[frame_col] == frame_num:
                for box in boxes:
                    # gets pixel value of the position
                    x_pos = float(cols[x_col])*w
                    y_pos = float(cols[y_col])*h
                    # checks if the gaze location is within the box
                    if x_pos <= float(box[2]) and x_pos >= float(box[0]) and y_pos <= float(box[3]) and  y_pos >= float(box[1]):
                        in_box += 1
                        break
                total += 1
                percentage_in_box = (in_box/total)*100
                
    else:
        percentage_in_box = -1                
    return percentage_in_box
