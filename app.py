import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
model = tf.keras.models.load_model('model.h5')



#Favicon and Header
st.set_page_config(
        page_title='Mnemo Classifier                 ',
        page_icon="🤖"
        )


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



col1, col2, col3, col4, col5 = st.columns([1,1,5,1,1])


with col3:

    def teachable_machine_classification(img, uploaded_file):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = keras.models.load_model(uploaded_file)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = img
        # image = Image.open(img_name).convert('RGB')
        # image = cv2.imread(image)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        #print(prediction)
        return np.argmax(prediction)



    st.title("Mnemo Post Classifier")
    st.write("Take a screenshot from my instagram posts and see if my trained Ai model is right at classiyfing my style of post")
    img_title = Image.open("1234.png")    
    st.image(img_title)
        
    # file upload and handling logic
    uploaded_file = st.file_uploader("Upload your screenshot here", type="png")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    #image = Image.open(img_name).convert('RGB')
        st.image(image, caption='This is your image', use_column_width=True)
        st.write("")
        st.write("Classifying post... please wait")


        label = teachable_machine_classification(image, 'model.h5')

        if label == 1:
            st.write("This doens't look like a MNEMO post 👎 ")
        else:
            st.write("This is likely to be a MNEMO post 👍  ")
            
