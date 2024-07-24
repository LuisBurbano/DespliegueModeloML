import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Path del modelo preentrenado
MODEL_PATH = 'pkl_archivo.pkl'

# Nombres de las columnas de entrada
column_names = [
    'edad', 'empleador_tipo', 'peso', 'educacion', 'numero_educacion',
    'estado_civil', 'ocupacion', 'relacion', 'raza', 'sexo', 'ganancia',
    'perdida', 'horas_semana', 'pais_natal'
]

# Se recibe la entrada y el modelo, devuelve la predicción
def model_prediction(data, model):
    x_n_df = pd.DataFrame([data], columns=column_names)
    preds = model.predict(x_n_df)
    return preds

def main():
    model = None

    # Se carga el modelo
    if model is None:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">PREDICCIÓN DE DATOS DEMOGRÁFICOS Y DE EMPLEO</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Entrada de datos
    edad = st.text_input("Edad:")
    empleador_tipo = st.text_input("Tipo de Empleador:")
    peso = st.text_input("Peso:")
    educacion = st.text_input("Educación:")
    numero_educacion = st.text_input("Número de Educación:")
    estado_civil = st.text_input("Estado Civil:")
    ocupacion = st.text_input("Ocupación:")
    relacion = st.text_input("Relación:")
    raza = st.text_input("Raza:")
    sexo = st.text_input("Sexo:")
    ganancia = st.text_input("Ganancia:")
    perdida = st.text_input("Pérdida:")
    horas_semana = st.text_input("Horas por Semana:")
    pais_natal = st.text_input("País Natal:")

    # El botón de predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):
        data = [
            int(edad),
            empleador_tipo,
            int(peso),
            educacion,
            int(numero_educacion),
            estado_civil,
            ocupacion,
            relacion,
            raza,
            sexo,
            int(ganancia),
            int(perdida),
            int(horas_semana),
            pais_natal
        ]
        predictS = model_prediction(data, model)
        st.success(f'La predicción es: {predictS[0]}')

if __name__ == '__main__':
    main()
