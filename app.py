import streamlit as st

from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model



col1, col2, col3, col4, col5 = st.columns([1,1,5,1,1])




def teachable_machine_classification(img, weights_file):
    # Load the model
    model = load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability







with col3:
    st.header("Mnemo Image Classifier")
    st.write("Take a screenshot of an Instagram post and let my own AI model decide whether it is a mnemo post or not")
    st.write('Hello')

    #create a file uploader widget
    uploaded_file = st.file_uploader("Choose a brain MRI ...", type="png")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'keras_model.h5')
        if label == 0:
            st.write(f"This is likely a mnemo post")
        else:
            st.write("This is likely NOT a mnemo post")






