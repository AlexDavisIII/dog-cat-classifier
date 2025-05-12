import streamlit as st
import requests
from PIL import Image
import io

prediction_key = st.secrets["api"]["prediction_key"]
endpoint = st.secrets["api"]["endpoint"]
project_id = st.secrets["api"]["project_id"]
published_name = st.secrets["api"]["published_name"]

# Streamlit UI
st.title("Image Classifier: Dog or Cat")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    # Convert image to binary stream
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")

    # Send to Azure Custom Vision
    headers = {
        "Prediction-Key": prediction_key,
        "Content-Type": "application/octet-stream"
    }

    url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{published_name}/image"

    response = requests.post(url, headers=headers, data=image_bytes.getvalue())

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        top_prediction = predictions[0]
        st.success(f"Prediction: **{top_prediction['tagName']}** ({top_prediction['probability']*100:.2f}%)")
    else:
        st.error(f"Error: {response.status_code} – {response.text}")
