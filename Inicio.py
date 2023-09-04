from PIL import Image
import streamlit as st
import base64

im = Image.open('imagenes/loto.png')

st.set_page_config(
    page_title="Detector de posturas de Yoga", 
    page_icon=im,
    layout="centered")

st.title("Yoga Pose Estimation")

url_ceia = "https://lse.posgrados.fi.uba.ar/posgrados/especializaciones/inteligencia-artificial"

'''
La siguiente aplicación se realizó en el marco del trabajo final de la Carrera de especialización de 
[Inteligencia Artificial](%s) de la facultad de ingeniería de la Universidad de Buenos Aires
''' %url_ceia

'''
:blue[Autor: ***Juan Ignacio Ribet***]
'''


'''


'''

