 # Euan Chree
 # 1912490
 # penumonia detection gui 

import streamlit as st
import numpy as np
import os
from keras.utils import load_img, img_to_array
from keras.utils.layer_utils import count_params
from keras.models import Functional
from keras.models import load_model as load_keras_model
from keras.applications import vgg16, inception_v3, densenet, xception

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Class_names
class_names = ["NORMAL", "PNEUMONIA"]
models_folder_path = (os.getcwd() + "/Models/")

@st.cache_resource
def load_model(model_path):
    '''
    Function to load a model from a given path

    Parameters:
        model_path : Path to the model
    Returns:
        image : The loaded model.
    '''
    return load_keras_model(model_path)

def get_model_input_size(model_path, model):
    '''
    Function to get the input size of a chosen model.

    Parameters:
        model_path : Path to the chosen model.
        model : The model object.
    '''

    # Extracting the input size from the model
    if not model.layers[0].input_shape[1] == None:
        return (model.layers[0].input_shape[1], model.layers[0].input_shape[2])

    # Falling back to manual
    if "inception" in model_path or "xception" in model_path:
        return (299, 299)
    elif "vgg16" in model_path or "densenet" in model_path:
        return (224, 224)

def load_image(image_path, target_size):
    '''
    Function to load an image.

    Parameters:
        image_path : Path to the image.
        target_size : Size to crop the image to.
    Returns:
        image : The loaded image.
    '''
    # Loading and formatting image
    image = load_img(image_path, target_size = target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    return image

# Function to make a prediction
def get_prediction(model, model_path, image):
    '''
    Function to a prediction from an image and a model

    Parameters:
        model : The model object.
        model_path : Path to the chosen model.
        image : The image to be classified.
    '''
    
    # Preprossing the image for the different pretrained models
    if "inception" in model_path:
        image = inception_v3.preprocess_input(image)
    elif "vgg16" in model_path:
        image = vgg16.preprocess_input(image)
    elif "densenet" in model_path:
        image = densenet.preprocess_input(image)
    elif "xception" in model_path:
        image = xception.preprocess_input(image)
    
    # Making and returning the prediction
    return model.predict([image])[0]
    
def get_diagnosis(prediction):
    '''
    Function to get a dignosis from a prediction.

    Parameters:
        prediction : Prediction output from the model.
    '''

    # Formatting the input and getting a classification label
    #pred_string = class_names[int(np.where(prediction[0] > 0.5, 1, 0))]
    pred_string = class_names[int(np.argmax(prediction))]
    print("Predicted:",prediction,"to be:",pred_string,".")
    if (pred_string == "NORMAL"):
        st.success("Selected image has been predicted to be negative for pneumonia.", icon="✅")
        #return "Selected image has been predicted to be negative for pneumonia."
    elif (pred_string == "PNEUMONIA"):
        st.error("Selected image has been predicted to be positive for pneumonia.", icon="🚨")
        #return "Selected image has been predicted to be positive for pneumonia."

st.title("Computer Aided Pneumonia Detection System")
st.subheader("Euan Chree 1912490")
st.markdown("### Select a model and an image file")

# Model selection
selected_model_path = st.selectbox("Select model:", os.listdir(models_folder_path))
if not selected_model_path:
    st.error("Couldn't find any models!", icon='❌')
    st.stop()

# Adding the model folder path
selected_model_path = models_folder_path + selected_model_path

# Loading model
selected_model = load_model(selected_model_path)

# Information on the model
with st.expander("Additional Information"):
    # Weight metrics
    st.markdown("#### Weights")
    total_col, trainable_col, non_trainable_col = st.columns(3)
    total_col.metric("Total Weights:", f"{(count_params(selected_model.trainable_weights) + count_params(selected_model.non_trainable_weights)):,}")
    trainable_col.metric("Trainable Weights:", f"{count_params(selected_model.trainable_weights):,}")
    non_trainable_col.metric("Non Trainable:", f"{count_params(selected_model.non_trainable_weights):,}")
    # Layers
    st.markdown("#### Layers:")
    for layer in selected_model.layers:
        if isinstance(layer, Functional):
            for func_layer in layer.layers: st.write(func_layer.name, ": ", str(type(func_layer)).split(".")[-1].removesuffix("'>"))
        else:
            st.write(layer.name, ": ", str(type(layer)).split(".")[-1].removesuffix("'>"))


# Getting the target image size from the model
image_size = get_model_input_size(selected_model_path, selected_model)
print("Found Model Input Size of:", image_size)

# Select image
selected_image_path = st.file_uploader("Select an image:")

if not selected_image_path:
    st.info("Please select an image", icon='📷')
    st.stop()

selected_image = load_image(selected_image_path, image_size)
print("Found Image at:", selected_image_path, "Converted to size:", image_size)
# Displaying the selected image
st.markdown("### Selected image")
st.image(selected_image_path)

# Information on the image
with st.expander("Additional Information"):
    res_col, col_col, size_col = st.columns(3)
    image_resolution = str(selected_image.shape[1]) + "x" + str(selected_image.shape[2] )
    col_col.metric("Colour Channels:", str(selected_image.shape[3]))
    res_col.metric("Resolution:", image_resolution)
    size_col.metric("Size:", f"{selected_image_path.size / 1000:.4g} Kb")
# Predict button
if st.button("Predict Diagnosis"):
    # Get Prediction
    prediction = get_prediction(selected_model, selected_model_path, selected_image)
    # Get Diagnosis
    st.markdown("### Diagnosis")
    get_diagnosis(prediction)
    st.markdown("---")
    # Class Probabilty information
    st.markdown("### Class Probabilty")
    for index, label in enumerate(class_names):
        st.write((label.lower().title()))
        st.progress(int(prediction[index]*100))
    st.markdown("---")
    normal_col, pneumonia_col = st.columns(2)
    normal_col.metric("Normal", str(prediction[0]))
    pneumonia_col.metric("Pneumonia", str(prediction[1]))
    print(prediction)
    st.markdown("---")
    
    # Clear results button
    if st.button("Clear Results 🗑️"):
        st.stop()