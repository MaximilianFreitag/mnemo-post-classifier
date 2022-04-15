import streamlit as st

from PIL import Image, ImageOps
import numpy as np
#import tensorflow as tf

#from img_classification import teachable_machine_classification


col1, col2, col3, col4, col5 = st.columns([1,1,5,1,1])


with col3:
    st.header("Mnemo Image Classifier")
    st.write("Take a screenshot of an Instagram post and let my own AI model decide whether it is a mnemo post or not")
    

    #create a file uploader widget
    uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg, jpeg, png")
    
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
