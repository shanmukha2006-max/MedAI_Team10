import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="DTIS — AI Medical Imaging Demo", layout="wide")
st.title("DTIS — AI Medical Imaging Demo")

img = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if img:
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(img), caption="Uploaded Image")

    # Send request to FastAPI
    with col2:
        st.write("Predictions:")
        try:
            res = requests.post("http://127.0.0.1:8000/predict", files={"file": img.getvalue()})

            
            if res.ok:
                probs = res.json()["probabilities"]
                efficiency = sum(probs)/len(probs)*100

                st.metric("Efficiency %", f"{efficiency:.2f}")

                st.write("---")

                for i, p in enumerate(probs):
                    st.write(f"Finding {i+1}: {p:.3f}")
                    st.progress(p)

            else:
                st.error("Prediction failed. Is the API running?")
        except Exception as e:
            st.error(f"Server not found. Please run: uvicorn dtis.inference.api:app --reload --port 8000")
