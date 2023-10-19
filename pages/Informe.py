"""
Informe.py

DESCRIPTION: generate the yoga practice report visualization.

AUTHOR: Juan Ignacio Ribet
DATE: 18-Sep-2023
"""

import cv2
import streamlit as st
import pandas as pd
from PIL import Image

im = Image.open('pages/Data/loto.png')

st.set_page_config(
    page_title="Detector de posturas de Yoga",
    page_icon=im,
    layout="wide")

st.title("yoga pose detector")

st.sidebar.title('Resultados')

posture = st.radio(
    "Elige la postura",
    ["Todo", "downdog", "goddess", "tree", "warrior"])

# Delete the information in the report DataFrame
clean = st.sidebar.button('Limpiar datos')
if clean:
    report_df = pd.DataFrame(columns=['pose',
                                      'punto',
                                      'ang optimo',
                                      'ang medido medio'])
    report_df.to_csv('pages/Data/report.csv', index=False)


def report_images(report, poses):
    '''
    Print the angles of the yoga practice report into the example images to easily visualize the 
    results

    Parameters:
        report (pd.DataFrame): Yoga practice report.
        poses (list) : poses names to use

    Returns:
        Image with all the predetermined angle in the exact position so it can be easy to read.
    '''
    angle_coord_images = {
        'downdog': {
            11: [0.75, 0.55],
            12: [0.4, 0.73],
            13: [0.85, 0.73],
            14: [0.65, 0.91],
            23: [0.42, 0.28],
            24: [0.4, 0.35],
            25: [0.14, 0.45],
            26: [0.11, 0.6],
        },
        'warrior': {
            11: [0.7, 0.35],
            12: [0.2, 0.38],
            13: [0.7, 0.15],
            14: [0.25, 0.15],
            23: [0.76, 0.55],
            24: [0.2, 0.55],
            25: [0.75, 0.8],
            26: [0.15, 0.75],
        },
        'warrior_inv': {
            11: [0.7, 0.38],
            12: [0.2, 0.35],
            13: [0.7, 0.15],
            14: [0.25, 0.15],
            23: [0.72, 0.55],
            24: [0.2, 0.55],
            25: [0.75, 0.75],
            26: [0.15, 0.75],
        },
        'tree': {
            23: [0.81, 0.5],
            24: [0.3, 0.45],
            25: [0.7, 0.72],
            26: [0.35, 0.72],
        },
        'tree_inv': {
            23: [0.13, 0.50],
            24: [0.65, 0.45],
            25: [0.25, 0.72],
            26: [0.7, 0.72],
        },
        'goddess': {
            23: [0.77, 0.5],
            24: [0.23, 0.5],
            25: [0.81, 0.72],
            26: [0.20, 0.72],
        }
    }
    for pose in poses:

        path = 'pages\Data\images_display\\' + pose + '.png'
        image = cv2.imread(path)
        annotated_image = image.copy()
        df = report.loc[report['pose'].isin([pose])]

        incr_width = int(0.08*annotated_image.shape[1])

        for i in (sorted(df['punto'].unique())):

            ang_op = df['ang optimo'][df['punto'] == i].item()
            ang_med = df['ang medido medio'][df['punto'] == i].item()
            ang_best = df['mejor'][df['punto'] == i].item()

            tolerance = 15
            if ang_med in range(ang_op-tolerance, ang_op+tolerance):
                text_color = (0, 255, 0)
            else:
                text_color = (0, 0, 255)

            coord = (int(angle_coord_images[pose][i][0]*annotated_image.shape[1]),
                     int(angle_coord_images[pose][i][1]*annotated_image.shape[0]))

            cv2.rectangle(annotated_image, (int(coord[0]-incr_width*1.3), coord[1]-30),
                          (coord[0]-incr_width+260, coord[1]+10),
                          (100, 100, 100),
                          -1)
            cv2.putText(annotated_image, str(f'{i}'), (int(coord[0]-incr_width*1.2), coord[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.putText(annotated_image, str(f'{ang_op}/'), (coord[0]-incr_width, coord[1]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2) #(3, 152, 252)
            cv2.putText(annotated_image, str(f'{ang_med}/'), (coord[0], coord[1]),
                        cv2.FONT_HERSHEY_PLAIN, 2, text_color, 2)
            cv2.putText(annotated_image, str(ang_best), (coord[0]+incr_width, coord[1]),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            prev_i = i

        cv2.rectangle(annotated_image, (0, 0), (500, 40), (100, 100, 100),
                      -1)
        cv2.putText(annotated_image, 'Angulos:', (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(annotated_image, 'Optimo/', (150, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2) #(3, 152, 252)
        cv2.putText(annotated_image, 'medio/', (290, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(annotated_image, 'mejor', (410, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.image(annotated_image)


# Load the report DataFrame
angle_mesure = pd.read_csv('pages/Data/report.csv')
angle_mesure.sort_values(by=['pose'])

# If there is data in the report it will be transformed to be presented.
if angle_mesure.shape[0] == 0:
    '''No hay datos'''
else:
    # Calculate the diference between the optimal angle and the mean angle)
    angle_mesure['dif'] = abs(
        angle_mesure['ang optimo']-angle_mesure['ang medido medio'])
    # Add column 'mejor' where is going to be the best angle reached
    angle_mesure['mejor'] = 0
    reporte_final = pd.DataFrame(columns=['pose',
                                          'punto',
                                          'ang optimo',
                                          'ang medido medio'])

    # Calculate the best angle in relation with the reference angle for each yoga posture.
    for j in angle_mesure['pose'].unique():
        informe = angle_mesure[angle_mesure['pose'] == j].copy()
        for i in informe['punto'].unique():
            minimo = informe[informe['punto'] == i]['dif'].min()
            best_ang = informe[(informe['punto'] == i) & (
                informe['dif'] == minimo)]['ang medido medio'][:1].item()
            informe.loc[informe['punto'] == i, 'mejor'] = best_ang
        informe = informe.drop(columns='dif')
        informe = informe.groupby(by=['pose', 'punto']
                                  ).mean().astype(int).reset_index()
        reporte_final = pd.concat([reporte_final, informe], ignore_index=True)

    reporte_final = reporte_final.astype({'ang optimo': 'int32',
                                          'ang medido medio': 'int32',
                                          'mejor': 'int32'})
    for i in range(reporte_final.shape[0]):
        reporte_final.loc[reporte_final.index == i, 'punto'] = int(
            ''.join(list(reporte_final['punto'][i])[5:7]))
    # Final dataframe to use for the visulization
    # Informationin reporte_final dataframe: {  'pose': Name of the yoga posture,
    #                                           'punto': central point of the measure angle,
    #                                           'ang optimo': optimal angle,
    #                                           'ang medido medio': Mean angle,
    #                                           'mejor': Best angle}

    # Time report: Load the time report
    report_time = pd.read_csv('pages\Data\\report_time.csv')
    report_time = report_time.groupby(by=['pose']).sum().astype(int).reset_index()
    # Information in report_time DataFrame: {'pose': Name of the yoga posture,
    #                                       'time': total time for each pose in seconds}
    total_time = report_time['time'].sum()

    # If the user select a specific position it will show only its example image
    if posture != "Todo":
        suma = (reporte_final['pose'] == posture).sum()
        posture = [posture]
        if posture[0] in ['warrior', 'tree']:
            posture_inv = str(posture[0]) + '_inv'
            posture.append(posture_inv)

        if suma > 0:
            col1, col2 = st.columns([2, 1])

            if len(report_time['time'][report_time['pose'] == posture[0]]):

                with col1:
                    report_images(reporte_final, posture)

                with col2:
                    st.image(
                        "pages\Data\images_display\\Landmarks.png")
                    # Show the total time for each posture
                    time_pose = report_time['time'][report_time['pose'] == posture[0]].item(
                    )
                    st.header(f'Tiempo de la postura: {time_pose} segundos')
            else:
                '''
                ## No hay datos de la postura
                '''
        else:
            '''
            ## No hay datos de la postura
            '''
    # If the user select to see all the example image report at once
    else:
        st.header(f'Tiempo total de entrenamiento: {total_time} segundos')
        poses = []
        for pose in sorted(reporte_final['pose'].unique()):
            if len(report_time['time'][report_time['pose'] == pose]):
                poses.append(pose)

        report_images(reporte_final, poses)

        #st.table(reporte_final)

video_file = open('pages\Data\\video.mp4', 'rb')
video_bytes = video_file.read()

if video_bytes:
    st.video(video_bytes)
