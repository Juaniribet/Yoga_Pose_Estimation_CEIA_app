import streamlit as st
from PIL import Image

im = Image.open('pages/Data/loto.png')

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

:blue[Director: ***Juan Pablo Pizarro (Globant)***]
'''

'''
# Resumen

Esta aplicación consiste en un detector de postura de yoga o asanas desarrollado para la empresa Globant. 
El proyecto pretende obtener el mínimo producto viable de una aplicación para entrenamiento de yoga, 
en donde se detecte la postura mediante algoritmos de aprendizaje de máquina y visión por computadora, 
y se le de soporte al usuario para realizarlas de forma correcta.

Para su desarrollo fueron fundamentales los conocimientos adquiridos en la carrera, en especial los 
vistos en las materias de visión por computadora con el uso indispensable de la librería OpenCV, 
como también los conceptos de aprendizaje de máquina, análisis de datos y la comprensión del mecanismo 
de funcionamiento de las de las redes neuronales convolucionales.

'''

'''
#### La aplicación tiene la posibilidad de detectar las siguientes 4 posturas
'''


col1, col2 = st.columns(2)

col3, col4 = st.columns(2)

with col1:
   st.header("Perro boca abajo")
   st.image("pages\Data\images_display\downdog.png")

with col2:
   st.header("Diosa")
   st.image("pages\Data\images_display\goddess.png")

with col3:
   st.header("Árbol")
   st.image("pages\Data\images_display\Tree.png")

with col4:
   st.header("Guerrero")
   st.image("pages\Data\images_display\warrior.png")