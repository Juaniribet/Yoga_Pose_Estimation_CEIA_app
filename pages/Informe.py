import streamlit as st
import pandas as pd
from PIL import Image

im = Image.open('pages/Data/imagenes/loto.png')

st.set_page_config(
    page_title="Detector de posturas de Yoga",
    page_icon=im,
    layout="wide")

st.title("yoga pose detector")

st.sidebar.title('Resultados')

genre = st.radio(
    "Elige la postura",
    ["Todo", "Downward_Facing_Dog", "goddess", "tree", "warrior"])


clean = st.sidebar.button('Limpiar datos')

if clean:
    report_df = pd.DataFrame(columns=['pose',
                                      'punto',
                                      'ang optimo',
                                      'ang medido medio'])
    report_df.to_csv('pages/Data/report.csv', index=False)

angle_mesure = pd.read_csv('pages/Data/report.csv')

if angle_mesure.shape[0] == 0:
    '''No hay datos'''
else:
    angle_mesure['dif'] = abs(
        angle_mesure['ang optimo']-angle_mesure['ang medido medio'])
    angle_mesure['mejor'] = 0
    reporte_final = pd.DataFrame(columns=['pose',
                                          'punto',
                                          'ang optimo',
                                          'ang medido medio'])

    for j in angle_mesure['pose'].unique():
        informe = angle_mesure[angle_mesure['pose'] == j].copy()
        # Calculate the best angle in relation with the reference angle.
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

    if genre != "Todo":
        suma = (reporte_final['pose'] == genre).sum()
        genre = [genre]
        if genre[0] in ['warrior', 'tree']:
            genre_inv = str(genre[0]) + '_inv'
            genre.append(genre_inv)

        if suma > 0:
            st.table(reporte_final.loc[reporte_final['pose'].isin(genre)])

            col1, col2 = st.columns([2, 1])

            with col1:
                path = "pages\Data\imagenes_mostrar\\informe\\" + \
                    genre[0] + ' - informe.png'
                st.image(path)

            with col2:
                st.image("pages\Data\imagenes_mostrar\\informe\\Landmarks.png")
        else:
            '''
            ## No hay datos de la postura
            '''

    else:
        st.table(reporte_final)
        col1, col2 = st.columns(2)

        col3, col4 = st.columns(2)

        poses = ['warrior', 'tree', 'goddess', 'Downward_Facing_Dog']

        for j, i in zip(poses, [col1, col2, col3, col4]):
            if (reporte_final['pose'] == j).sum() > 0:
                with i:
                    path = "pages\Data\imagenes_mostrar\\informe\\" + j + ' - informe.png'
                    st.image(path)

video_file = open('pages\Data\\video.mp4', 'rb')
video_bytes = video_file.read()

if video_bytes:
    st.video(video_bytes)
