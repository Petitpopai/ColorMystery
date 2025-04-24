"""
Streamlit UI – ColorNumber avec curseur de simplification
"""
import io, json, tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from core_simple import generate_number_sheet

TEXT = {
    "difficulty": "Difficulty",
    "simplify": "Simplification level",
    "width": "Resize (px)",
    "upload": "Drag & drop an image",
    "preview": "Preview",
    "process": "Generate",
    "running": "Processing..."
}
_ = lambda k: TEXT[k]

st.set_page_config(page_title="ColorNumber", layout="wide")
st.title("ColorNumber – generate your colour-by-number sheets")

col1, col2 = st.columns(2)
with col1:
    diff = st.selectbox(_("difficulty"), ["easy", "medium", "hard"], index=1)
with col2:
    simp = st.selectbox(_("simplify"), ["low", "medium", "high"], index=1)

width = st.number_input(_("width"), 256, 4096, 1024, 64)
uploaded = st.file_uploader(_("upload"), type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded, caption=_("preview"), use_column_width=True)
    if st.button(_("process")):
        with st.spinner(_("running")):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(uploaded.read()); tmp.flush()
            img = Image.open(tmp.name).convert("RGBA")
            k = {"easy": 8, "medium": 16, "hard": 24}[diff]
            sheet = generate_number_sheet(img, k, simp, "medium")

        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        st.image(sheet, use_column_width=True)
        st.download_button("⬇️ Download sheet.png", data=buf.getvalue(),
                           file_name="sheet.png", mime="image/png")
