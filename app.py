import streamlit as st
import requests
from PIL import Image
import io


# === SET THESE VARIABLES FROM YOUR AZURE PROJECT ===
prediction_key = "7S4G8hy2Jxm5IxVOx2W0EDl9C1rdRWeXsEoVzoPMgA6cCXxuh2juJQQJ99BEACHYHv6XJ3w3AAAIACOGlQm4"
endpoint = "https://dogcatclassifier-prediction.cognitiveservices.azure.com"  # e.g., https://<region>.api.cognitive.microsoft.com
project_id = "3e9cb624-99f1-44a7-b577-92b8191abfb9"
published_name = "Iteration1"

# Streamlit UI
st.title("Custom Image Classifier")

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
