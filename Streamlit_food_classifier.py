import streamlit as st
from fastai.vision.all import *
import gdown

st.markdown("""# Mongolian Food Classifier

There are many traditional food's in Mongolia, This classifier allows you to upload your own picture of Mongolian food, and it will classify 
as one of the following, which are Buuz, Tsuivan, Khuushuur, Niisel Salad aka Potato Salad. Try it out and see for yourself.""")

st.markdown("""### Upload your food image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

## Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1XpIo1mcxbtHSNq2-g9VVpG37mlyW17HS'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    with col1:
        st.markdown(f"""### Predicted Food: {pred.capitalize()}""")
        st.markdown(f"""### Accuracy: {round(max(probs.tolist()), 3) * 100}%""")
    with col2:
        st.image(img, width=300)