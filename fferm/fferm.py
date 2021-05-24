# -*- coding: utf-8 -*-
import os

import cv2
import face_recognition
import fer
import numpy as np
import pandas as pd
from fer import FER


def index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]


def overlap_index(l):
    return index_multi(l, [x for x in set(l) if l.count(x) > 1][0])


def fferm(movie_path, output_path, freq=1, resize=1.0, threshold=0.45, info=True, face_exp=1.0):
    detector = FER(mtcnn=False)
    
    cwd = os.getcwd()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if not os.path.exists(os.path.join(output_path, 'fr_index')):
        os.mkdir(os.path.join(output_path, 'fr_index'))

    os.chdir(os.path.join(output_path, 'fr_index'))
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0
    output = []
    name_num = 1  # name index
    known_face_names = []  
    known_face_encodings = []  

    # Open the input movie file
    input_movie = cv2.VideoCapture(movie_path)
    fps = input_movie.get(cv2.CAP_PROP_FPS)
    img_height = input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize
    img_width = input_movie.get(cv2.CAP_PROP_FRAME_WIDTH) * resize
    frames = input_movie.get(cv2.CAP_PROP_FRAME_COUNT)
    freq_f = round(freq * fps)

    while True:
        input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = input_movie.read()
        if not ret:
            break
        minutes = int((frame_number/fps)//60)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        small_frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
        rgb_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)

        # detect emotions(using FER) from img and face_locations(from fr)
        face_rectangles = []
        for (top, right, bottom, left) in face_locations:
            face_width = right - left  # face_expを掛ける前
            face_height = bottom - top
            x = int((left-(face_exp-1.0)*face_width*0.5))
            y = int((top-(face_exp-1.0)*face_height*0.5))
            w = int(face_width*face_exp)
            h = int(face_height*face_exp)
            face_rectangles.append((x, y, w, h))

        results = detector.detect_emotions(rgb_frame, face_rectangles)
        fes = []
        for result in results:
            fe = []
            fe.append(result['emotions']['angry'])
            fe.append(result['emotions']['disgust'])
            fe.append(result['emotions']['fear'])
            fe.append(result['emotions']['happy'])
            fe.append(result['emotions']['neutral'])
            fe.append(result['emotions']['sad'])
            fe.append(result['emotions']['surprise'])
            fes.append(fe)

        face_names = []  
        all_distances = []
        per_frame = []

        for (top, right, bottom, left), face_encoding, fe in zip(face_locations, face_encodings, fes):
            if len(known_face_encodings) == 0:
                name = 0  # 0="Unknown"
                face_names.append(name)
            else:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=threshold)
                name = 0  # "Unknown"
            # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding).tolist()
                best_match_index = face_distances.index(min(face_distances))
                all_distances.append(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    face_names.append(name)
                    no_unknown = ([s for s in face_names if s != 0])
                    while len(no_unknown) != len(set(no_unknown)):
                        ol_face = no_unknown[overlap_index(no_unknown)[
                            0]]
                        ol_index = index_multi(
                            face_names, ol_face)
                        ol_name_index = known_face_names.index(
                            ol_face)
                        if all_distances[ol_index[0]][ol_name_index] > all_distances[ol_index[1]][ol_name_index]:
                            all_distances[ol_index[0]][ol_name_index] = 1.0
                            if len([x for x in all_distances[ol_index[0]] if x < threshold]) == 0:
                                face_names[ol_index[0]] = 0
                            else:
                                face_names[ol_index[0]] = known_face_names[all_distances[ol_index[0]].index(
                                    min(all_distances[ol_index[0]]))]
                        else:
                            all_distances[ol_index[1]][ol_name_index] = 1.0
                            if len([x for x in all_distances[ol_index[1]] if x < threshold]) == 0:
                                face_names[ol_index[1]] = 0
                            else:
                                face_names[ol_index[1]] = known_face_names[all_distances[ol_index[1]].index(
                                    min(all_distances[ol_index[1]]))]
                        no_unknown = ([s for s in face_names if s != 0])
                else:
                    face_names.append(name)
            face_width = right - left
            face_height = bottom - top
            to_time = round(input_movie.get(cv2.CAP_PROP_POS_MSEC))
            to_x = (left-(face_exp-1.0)*face_width*0.5)/img_width
            to_y = (top-(face_exp-1.0)*face_height*0.5)/img_height
            to_w = face_width*face_exp/img_width
            to_h = face_height*face_exp/img_height
            per_face = [to_time,
                        to_x,
                        to_y,
                        to_w,
                        to_h]
            per_face.extend(fe)
        for unknown_id in index_multi(face_names, 0):
            known_face_encodings.append(face_encodings[unknown_id])
            known_face_names.append(name_num)
            face_names[unknown_id] = name_num
            top, right, bottom, left = face_locations[unknown_id]
            img_to_w = small_frame[top: bottom, left: right]
            cv2.imwrite('{}.jpg'.format(name_num), img_to_w)
            name_num += 1
        for x, y in zip(per_frame, face_names):
            x.append(y)
        output.extend(per_frame)
        frame_number += round(freq_f)
    # All done!
    input_movie.release()
    os.chdir(cwd)
    return output, known_face_encodings
