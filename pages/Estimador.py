"""
Estimador.py

DESCRIPTION: generate the pose estimatios.

AUTHOR: Juan Ignacio Ribet
DATE: 08-Sep-2023
"""

import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import math


im = Image.open('pages/Data/loto.png')

st.set_page_config(
    page_title="Detector de posturas de Yoga",
    page_icon=im,
    layout="wide")

st.title("yoga pose detector")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria_expanded="true"] > dic:first-child{
        width:350x
    }

    [data-testid="stSidebar"][aria_expanded="false"] > dic:first-child{
        width:350x
        margin-left: -350x
    }
    </style>
    """,
    unsafe_allow_html=True,
)

"""
precionar el boton "Iniciar" para empezar al estimación de postura en el entrenamiento.

Al finalizar precionar el boton "Fin"

Si quiere realizar la estimacion en el video de ejemplo precionar el boton "video demostración"
"""


col1, col2 = st.columns([1, 5])
with col1:
    run = st.button('Iniciar')
with col2:
    stop = st.button("Fin")

frame_placeholder = st.empty()
placeholder = st.empty()
placeholder2 = st.empty()

def camera_relese():
    try:
        cap.release()
    except:
        pass

# Select the camera
camera = st.sidebar.number_input(
    'seleccionar camara', value=0, max_value=2, on_change=camera_relese)

save = st.sidebar.checkbox('Guardar video')

demo_video = st.sidebar.button('video demostración')

if demo_video:
    run = True

with open('pages\Data\models\yoga_pose_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_pose = mp.solutions.pose  # Mediapipe Solutions
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers



def inFrame(lst):
    '''
    Check if specific landmarks in a list of facial landmarks are visible.

    This function checks the visibility confidence of specific body landmarks (landmark indices
    15, 16, 27, and 28) and returns True if at least one of the landmarks on each side of the face
    is visible with a confidence greater than 0.6.

    Parameters:
        lst (list): A list containing facial landmarks, where each landmark is represented as an
                    object with attributes like 'visibility'.

    Returns:
        bool: True if at least one landmark on each side of the face is visible with confidence
              greater than 0.6, False otherwise.
    '''
    if ((lst[28].visibility > 0.6 or lst[27].visibility > 0.6)
            and (lst[15].visibility > 0.6 or lst[16].visibility > 0.6)):
        return True
    return False


def calculate_angle_coord(p_cood_list):
    """
    Calculate the angle formed by three coordinates in a 2D plane.

    Parameters:
        p_cood_list (list): A list containing three 2D coordinate points as numpy arrays.

    Returns:
        float: The angle in degrees between the lines connecting the first and second points
               and the second and third points. The angle is always in the range [0, 180].
    
    first_point = p_cood_list[0][*]
    mid_point = p_cood_list[1][*]
    last_point = p_cood_list[2][*]
    """
    radians = math.atan2(p_cood_list[2][1]-p_cood_list[1][1], p_cood_list[2][0]-p_cood_list[1][0]) - \
        math.atan2(p_cood_list[0][1]-p_cood_list[1][1], p_cood_list[0][0]-p_cood_list[1][0])
    angle = abs(radians*180.0/math.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


landmarks = []
for val in range(0, 33):
    landmarks += ['x{}'.format(val), 
                  'y{}'.format(val),
                  'z{}'.format(val), 
                  'v{}'.format(val)]


# Define the angles to messure
dic = {'downdog': {(12, 24, 26): 56, 
                    (24, 26, 28): 174, 
                    (24, 12, 14): 169,
                    (12, 14, 16): 165, 
                    (11, 23, 25): 57, 
                    (23, 25, 27): 174,
                    (23, 11, 13): 169, 
                    (11, 13, 15): 167},
        'warrior': {(16, 14, 12): 180, 
                    (11, 13, 15): 180, 
                    (14, 12, 24): 90, 
                    (13, 11, 23): 90,
                    (12, 24, 26): 140, 
                    (11, 23, 25): 120, 
                    (24, 26, 28): 170, 
                    (23, 25, 27): 120},
        'warrior_inv': {(16, 14, 12): 180, 
                        (11, 13, 15): 180, 
                        (14, 12, 24): 90, 
                        (13, 11, 23): 90,
                        (12, 24, 26): 120, 
                        (11, 23, 25): 140, 
                        (24, 26, 28): 120, 
                        (23, 25, 27): 170},
        'goddess': {(12, 24, 26): 120, 
                    (11, 23, 25): 120, 
                    (24, 26, 28): 120, 
                    (23, 25, 27): 120},
        'tree': {(12, 24, 26): 170, 
                (11, 23, 25): 120, 
                (24, 26, 28): 170, 
                (23, 25, 27): 60},
        'tree_inv': {(12, 24, 26): 120, 
                    (11, 23, 25): 170, 
                    (24, 26, 28): 60, 
                    (23, 25, 27): 170}
        }


# Images to display on screen
dic_images = {'downdog': 'pages/Data/images_display/video/downdog.png',
                'warrior': 'pages/Data/images_display/video/warrior.png',
                'warrior_inv': 'pages/Data/images_display/video/warrior_inv.png',
                'goddess': 'pages/Data/images_display/video/goddess.png',
                'tree': 'pages/Data/images_display/video/Tree.png',
                'tree_inv': 'pages/Data/images_display/video/tree_inv.png'
                }

if run:
    if demo_video:
        cap = cv2.VideoCapture("pages\Data\\sample_video.mp4")
        cap.set(cv2.CAP_PROP_FPS,5)
    else:
        cap = cv2.VideoCapture(camera)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    result = cv2.VideoWriter('pages/Data/video.mp4', 
                             cv2.VideoWriter_fourcc(*'VIDX'),
                             20, 
                             (frame_width, frame_height))




    if not cap.isOpened():
        st.sidebar.text(f'La camara {camera} no está disponible')
        cap = cv2.VideoCapture(0)

    # Iniciate variables
    body_language = 0
    body_language_time = 0
    pose_time = 0
    report = []
    report_time = []
    prev_frame_time = 0
    new_frame_time = 0

    start_time = time.time()

    with mp_pose.Pose(model_complexity=1, 
                      smooth_landmarks = True, 
                      min_detection_confidence=0.5, 
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if stop:
                break
            if not success:
                st.sidebar.write('the video capture end')
                break
            if frame is None:
                break

            # Mido FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time    
            fps = int(fps)

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Export coordinates
            try:
                circle_coord = ((frame_width-100), 40)

                if results.pose_landmarks and inFrame(results.pose_landmarks.landmark):

                    # Draw the landmarks connections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                              mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                     thickness=1,
                                                                     circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(219, 170, 117),
                                                                     thickness=1,
                                                                     circle_radius=2)
                                              )

                    # Extract Pose landmarks
                    poses = results.pose_landmarks.landmark
                    row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                         for landmark in poses]).flatten())

                    # Make Detections every 3 seconds
                    time_laps = 3
                    current_time = int(time.time()-start_time)
                    if (body_language == 0) or (current_time % time_laps == 0):
                        X = pd.DataFrame([row], columns=landmarks)
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]


                    if body_language_prob[np.argmax(body_language_prob)] < 0.50:
                        body_language = 0

                    # Draw the status box
                    cv2.rectangle(image,
                                  (0, 0),
                                  (600, 60),
                                  (0, 0, 0),
                                  -1)

                    # Display pose detected
                    cv2.putText(image,
                                'POSE',
                                (195, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (250, 250, 250),
                                1,
                                cv2.LINE_AA)

                    # Check if the pose detection probability is greater than 80%
                    if body_language_prob[np.argmax(body_language_prob)] >= 0.80:
                        
                        body_language = body_language_class

                        # record the time for each posture.
                        if (body_language != 0) & (body_language != body_language_time):
                            finish_pose_time = int(time.time()-pose_time)
                            pose_time = time.time()
                            if body_language_time != 0:
                                report_time.append([body_language_time,finish_pose_time])

                            body_language_time = body_language
                        
                        
                        cv2.putText(image,
                                    body_language_class.split(' ')[0],
                                    (190, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)
                        cv2.circle(image, 
                                   circle_coord, 
                                   40, 
                                   (0, 255, 0), 
                                   -1)
                    else:
                        cv2.circle(image,
                                   circle_coord,
                                   40,
                                   (0, 0, 255),
                                   -1)

                    # Display Probability
                    cv2.putText(image,
                                'PROB',
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (250, 250, 250),
                                1,
                                cv2.LINE_AA)
                    cv2.putText(image,
                                f'{body_language_prob[np.argmax(body_language_prob)]:.0%}',
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA)

                    if body_language:
                        # The poses 'warrior' and 'tree' are not symmetrical. The next lines check  
                        # the angle in P26 to check whether the pose detected is rigth or left
                        # and select the angles to measure in correspondence.
                        dato_pose = body_language_class
                        if body_language_class in ['warrior', 'tree']:
                            angle_1 = 0
                            try:
                                angle_1 = int(calculate_angle_coord([(
                                    results.pose_landmarks.landmark[24].x,
                                    results.pose_landmarks.landmark[24].y),
                                    (
                                    results.pose_landmarks.landmark[26].x,
                                    results.pose_landmarks.landmark[26].y),
                                    (
                                    results.pose_landmarks.landmark[28].x,
                                    results.pose_landmarks.landmark[28].y)]))

                            except:
                                pass
                            if angle_1 < 150:
                                class_inv = body_language_class + '_inv'
                                p_dic = dic.get(class_inv)
                                dato_pose = class_inv
                            else:
                                p_dic = dic.get(body_language_class)

                        else:
                            p_dic = dic.get(body_language_class)
                        
                        
                        q_ang = 0 # inicilice the counting of the angles measured
                        q_ang_ok = 0 # inicilice the counting of the angles measured ok

                        # Extract the angles points to measure from angle dict.
                        for i in range(len(p_dic)):
                            p_cood_list = []
                            midle_point = list(p_dic.keys())[i][1]
                            for p in list(p_dic.keys())[i]:
                                if results.pose_landmarks.landmark[p].visibility > 0.5:
                                    
                                    p_corrd = (
                                        results.pose_landmarks.landmark[p].x,
                                        results.pose_landmarks.landmark[p].y)
                                    p_cood_list.append(p_corrd)
                                else:
                                    break

                            try:
                                # Meassure the angles
                                angle = int(calculate_angle_coord(p_cood_list))
                                angle_ok = int(list(p_dic.values())[i])
                                q_ang += 1
                                report.append([dato_pose,
                                               list(p_dic.keys())[i],
                                               angle_ok,
                                               angle])

                                # Print the angles into the image. Green if it is between the 
                                # tolerance and red if it is not.
                                tolerance = 15
                                if angle in range(angle_ok-tolerance, angle_ok+tolerance):
                                    q_ang_ok += 1
                                    text_color = (0, 255, 0)
                                else:
                                    text_color = (0, 0, 255)

                                cv2.putText(image,
                                            str(angle),
                                            (int((results.pose_landmarks.landmark[midle_point].x) \
                                                 *image.shape[1]),
                                             int((results.pose_landmarks.landmark[midle_point].y) \
                                                 *image.shape[0])),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            2,
                                            text_color,
                                            2)
                                cv2.putText(image,
                                            f'({angle_ok})',
                                            (int((results.pose_landmarks.landmark[midle_point].x) \
                                                 *image.shape[1]),
                                             int((results.pose_landmarks.landmark[midle_point].y) \
                                                 *image.shape[0])+25),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            1,
                                            (66, 245, 236),
                                            1)

                            except:
                                pass

                        # Insert the example picture into the image
                        img_path = dic_images.get(dato_pose)
                        img = cv2.imread(img_path)
                        h = img.shape[0]
                        w = img.shape[1]
                        image[image.shape[0]-(h+10):image.shape[0]-10,
                                image.shape[1]-(w+10):image.shape[1]-10] = img
                        
                        # if any angle is out of range a red led turns on
                        if q_ang_ok != q_ang:
                            cv2.circle(image,
                               circle_coord,
                               20,
                               (0, 0, 250),
                               -1)

                # Print if the body is not fully visible                    
                else:
                    cv2.putText(image,
                                "Make your Full",
                                (50, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (66, 245, 236),
                                3)
                    cv2.putText(image,
                                "body visible",
                                (50, 65),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (66, 245, 236),
                                3)
                    cv2.circle(image,
                               circle_coord,
                               40,
                               (0, 0, 255),
                               -1)

            except:
                pass
   
            cv2.putText(image, 
                        f'FPS: {str(fps)}', 
                        (15, frame_height-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1 , 
                        (250, 250, 250),
                        2)
            if save:
                result.write(image)                    
            frame_placeholder.image(image, channels='BGR')
                            
            # Show the time 
            placeholder.text(f'time: {int(time.time()-start_time)}')          

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Save the results
    report_df = pd.DataFrame(report, columns=['pose',
                                                'punto',
                                                'ang optimo',
                                                'ang medido medio'])
    report_df.to_csv(
        'pages/Data/report.csv', index=False)
    
    finish_pose_time = int(time.time()-pose_time)
    
    # Save last posture time
    pose_time = time.time()
    report_time.append([body_language_time,finish_pose_time])
    report_time_df = pd.DataFrame(report_time, columns=['pose','time'])
    report_time_df.to_csv(
            'pages/Data/report_time.csv', index=False)  

    result.release()            
    cap.release()
    cv2.destroyAllWindows()
