import streamlit as st
import pickle
import pandas as pd
import numpy as np



# Cargar el modelo desde el archivo pickle
with open('../models/best_naive_bayes.pk', 'rb') as file:
    modelo = pickle.load(file)


# Cargar el modelo desde el archivo pickle
with open('../models/count_vectorizer_for_naive_bayes.pk', 'rb') as file:
    count_vector = pickle.load(file)


def predecir_sentimiento(texto):

    prueba = count_vector.transform([texto]).toarray()
    prediccion = modelo.predict(prueba)[0]

    return prediccion



# Configurar la interfaz de la aplicación con Streamlit
def main():
    st.title('Sentiment Classifier with Naive Bayes')
    
    st.write('This classifier uses a Naive Bayes model with 82% accuracy in classifying English texts as positive or negative.')

    st.write("The data used to train the model are 890 reviews from the Google Play Store so please don't take the model's capacity too much into consideration with real sentences.")


    # Cuadro de texto para introducir el texto
    texto_usuario = st.text_area('Enter your text here:', placeholder='i love this // i hate everithing')
    
    if st.button('Predict feeling'):
        if texto_usuario:
            # Realizar la predicción
            sentimiento = predecir_sentimiento(texto_usuario)
            if sentimiento == 0:
                st.warning('This text is negative! >:( ')
            if sentimiento == 1:
                st.success('This text is positive! ;) ')
        else:
            st.info('Please enter a text to make the prediction.')

if __name__ == '__main__':
    main()
