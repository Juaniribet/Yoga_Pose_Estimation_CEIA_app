import streamlit as st
import pandas as pd
import numpy as np

#im = Image.open('Interfaz\imagenes\loto.png')

st.set_page_config(layout="wide")

st.title("yoga pose detector")

st.sidebar.title('Resultados')

clean = st.sidebar.button('Limpiar datos')

if clean:
    report_df = pd.DataFrame(columns=['pose',
                                    'punto',
                                    'ang optimo',
                                    'ang medido medio'])
    report_df.to_csv('pages/Data/report.csv', index=False)

placeholder = st.empty()
angle_mesure= pd.read_csv('pages/Data/report.csv')

if angle_mesure.shape[0] == 0:
    placeholder.text('No hay datos')
else:
    angle_mesure['dif'] = abs(angle_mesure['ang optimo']-angle_mesure['ang medido medio'])
    angle_mesure['mejor'] = 0

    reporte_final = pd.DataFrame(columns=['pose', 'punto', 'ang optimo', 'ang medido medio'])
    for j in angle_mesure['pose'].unique():
        repo = angle_mesure[angle_mesure['pose'] == j].copy()
        for i in repo['punto'].unique():
            minimo = repo[repo['punto'] == i]['dif'].min()
            best_ang = repo[(repo['punto'] == i) & (repo['dif'] == minimo)]['ang medido medio'][:1].item()
            repo.loc[repo['punto'] == i, 'mejor'] = best_ang
        repo = repo.drop(columns='dif')
        informe = repo.groupby(by=['pose','punto']).mean().astype(int).reset_index()
        reporte_final = pd.concat([reporte_final,informe],ignore_index=True)
    reporte_final = reporte_final.astype({'ang optimo': 'int32', 'ang medido medio': 'int32', 'mejor': 'int32'})

    st.table(reporte_final)



