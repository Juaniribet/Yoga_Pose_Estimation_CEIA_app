import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

#im = Image.open('Interfaz\imagenes\loto.png')

st.set_page_config(layout="wide")

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

# st.sidebar.title('opciones')
# st.sidebar.subheader('parameters')

col1, col2= st.columns(2)
with col1:
    run = st.button('Run')
with col2:
    stop = st.button("Stop")

frame_placeholder = st.empty()
placeholder = st.empty()
placeholder2 = st.empty()


# change_camera = st.sidebar.button("check camera avaliable")

# if change_camera:
#     for i in range(2):
#         cap = cv2.VideoCapture(i)
#         test, frame = cap.read()
#         if test:
#             st.sidebar.text(f'camera: {i}')

#camera = st.sidebar.number_input('select camera', value = 0, min_value=0, max_value=10)



with open('pages/Data/yoga_pose_detection.pkl', 'rb') as f:
    model = pickle.load(f)

mp_pose = mp.solutions.pose  # Mediapipe Solutions
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
scaler = StandardScaler()

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
    """
    first_point = np.array(p_cood_list[0])  # First
    mid_point = np.array(p_cood_list[1])  # Mid
    last_point = np.array(p_cood_list[2])  # End

    radians = np.arctan2(last_point[1]-mid_point[1], last_point[0]-mid_point[0]) - \
        np.arctan2(first_point[1]-mid_point[1], first_point[0]-mid_point[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

landmarks = []
for val in range(0, 33):
    landmarks += ['x{}'.format(val), 'y{}'.format(val),
                  'z{}'.format(val), 'v{}'.format(val)]

def camera_relese():
    try:
        cap.release()
    except:
        pass

# Select the camera
camera = st.sidebar.number_input('select camera',value =0, max_value =2, on_change=camera_relese)

if camera:
    run = True


if run:
    # Define the angles to messure
    dic = {'Downward_Facing_Dog': {(12, 24, 26): 56, (24, 26, 28): 174, (24, 12, 14): 169, 
                                   (12, 14, 16): 165, (11, 23, 25): 57, (23, 25, 27): 174, 
                                   (23, 11, 13): 169, (11, 13, 15): 167},
       'warrior': {(16, 14, 12): 180, (11, 13, 15): 180, (14, 12, 24): 90, (13, 11, 23): 90, 
                   (12, 24, 26): 140, (11, 23, 25): 120, (24, 26, 28): 170, (23, 25, 27): 120},
       'goddess': {(12, 24, 26): 120, (11, 23, 25): 120, (24, 26, 28): 120, (23, 25, 27): 120},
       'tree': {(12, 24, 26): 170, (11, 23, 25): 120, (24, 26, 28): 170, (23, 25, 27): 60}
       }

    # Images to show
    dic_images = {'Downward_Facing_Dog': 'pages/Data/imagenes_mostrar/Downward_Facing_Dog.png',
                'warrior': 'pages/Data/imagenes_mostrar/warrior.png',
                'goddess': 'pages/Data/imagenes_mostrar/goddess.png',
                'tree': 'pages/Data/imagenes_mostrar/Tree.png'
                }

        
    cap = cv2.VideoCapture(camera)
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    
    if not cap.isOpened():
        st.sidebar.text(f'La camara {camera} no está disponible')
        cap = cv2.VideoCapture(0)
    


    # Iniciate variables
    body_language = 0
    report = []
    # report_df = pd.DataFrame(report, columns=['pose',
    #                                           'punto',
    #                                           'ang optimo',
    #                                           'ang medido medio'])
    # report_df.to_csv('pages/Data/report.csv', index=False)

    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, frame = cap.read()
            if stop:
                break
            if not success:
                st.sidebar.write('the video capture end')
                break
            if frame is None:
                break

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
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), 
                                                                     thickness=2, 
                                                                     circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(245, 66, 230),
                                                                       thickness=2,
                                                                       circle_radius=2)
                                            )

                    # Extract Pose landmarks
                    poses = results.pose_landmarks.landmark
                    row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                        for landmark in poses]).flatten())

                    # # Concate rows
                    # row = pose_row

                    # Make Detections every 5 seconds
                    time_laps = 5
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
                                  (245, 117, 16),
                                  -1)

                    # Display pose
                    cv2.putText(image,
                                'POSE',
                                (195, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA)

                    if body_language_prob[np.argmax(body_language_prob)] >= 0.80:
                        body_language = body_language_class
                        cv2.putText(image,
                                    body_language_class.split(' ')[0],
                                    (190, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)
                        cv2.circle(image, circle_coord, 40, (0, 255, 0), -1)
                    else:
                        cv2.putText(image,
                                    "",
                                    (190, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)
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
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA)
                    cv2.putText(image,
                                str(round(body_language_prob[np.argmax(body_language_prob)], 3)),
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA)

                    if body_language:
                        w = 400
                        h = 310
                        img_path = dic_images.get(body_language_class)
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, [w, h])
                        img = cv2.flip(img, 1)

                        p_dic = dic.get(body_language_class)

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
                                angle = int(calculate_angle_coord(p_cood_list))
                                angle_ok = int(list(p_dic.values())[i])

                                report.append([body_language_class,
                                               list(p_dic.keys())[i],
                                               angle_ok,
                                               angle])
                                
                                # Save the results
                                report_df = pd.DataFrame(report, columns=['pose',
                                                      'punto',
                                                      'ang optimo',
                                                      'ang medido medio'])
                                report_df.to_csv('pages/Data/report.csv', index=False)

                                tolerance = 15
                                if angle in range(angle_ok-tolerance, angle_ok+tolerance):
                                    text_color = (0, 255, 0)
                                else:
                                    text_color = (0, 0, 255)
                                cv2.putText(image,
                                            str(angle),
                                            (int((results.pose_landmarks.landmark[midle_point].x)*image.shape[1]),
                                            int((results.pose_landmarks.landmark[midle_point].y)*image.shape[0])),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            3,
                                            text_color, 
                                            3)
                                cv2.putText(image,
                                            f'({angle_ok})',
                                            (int((results.pose_landmarks.landmark[midle_point].x)*image.shape[1]),
                                            int((results.pose_landmarks.landmark[midle_point].y)*image.shape[0])+25),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            2,
                                            (66, 245, 236),
                                            2)

                                image[image.shape[0]-(h+10):image.shape[0]-10,
                                    image.shape[1]-(w+10):image.shape[1]-10] = img
                            except:
                                pass

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

            frame_placeholder.image(image, channels='BGR') #, use_column_width = "always"
            placeholder.text(f'time: {int(time.time()-start_time)}')
            placeholder.text(report)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()




