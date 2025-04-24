# app.py  (version corrigée)
"""
Interface Streamlit – inclut case « Contours noirs sur fond blanc »
"""
import io, json, tempfile
from pathlib import Path
import streamlit as st
from core import stream_process

# --- i18n ---------------------------------------------------------------------------
LOCALES = {lang: json.loads(Path(__file__).with_name("i18n").joinpath(f"{lang}.json").read_text())
           for lang in ("en", "fr")}


def _t(key: str) -> str:
    return LOCALES[st.session_state.get("lang", "en")].get(key, key)


# --- barre latérale -----------------------------------------------------------------
st.set_page_config(page_title="ColorMystery", layout="wide")
with st.sidebar:
    st.selectbox("Language / Langue", ["en", "fr"], key="lang")
    st.title("ColorMystery")
    mode = st.selectbox(_t("mode"), ["both", "number", "mystery"])
    difficulty = st.radio(_t("difficulty"), ["easy", "medium", "hard"], index=1)
    detail = st.radio(_t("detail"), ["low", "medium", "high"], index=1)
    width = st.number_input(_t("width"), 256, 8192, 1024, 64)
    tile = st.number_input(_t("tile_size"), 256, 2048, 1024, 64)
    pure = st.checkbox("Contours noirs sur fond blanc", value=True)

# --- zone principale ----------------------------------------------------------------
uploaded = st.file_uploader(_t("upload"), type=["png", "jpg", "jpeg"])
if uploaded:
    st.image(uploaded, caption=_t("preview"), use_column_width=True)
    if st.button(_t("process")):
        with st.spinner(_t("running")):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(uploaded.read()); tmp.flush()

            class Args:  # mini‐namespace pour réutiliser stream_process
                pass
            Args.image = tmp.name
            Args.mode = mode
            Args.difficulty = difficulty
            Args.detail = detail
            Args.id_set = [str(i) for i in range(1, 1000)]
            Args.k = {"easy": 8, "medium": 16, "hard": 24}[difficulty]
            Args.width = width
            Args.tile_size = tile
            Args.pure_outline = pure

            sheets = stream_process(Path(tmp.name), Args)

        st.success("✓")
        for m, im in sheets.items():
            if mode in (m, "both"):
                st.subheader(m)
                st.image(im, use_column_width=True)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                st.download_button(
                    label=f"⬇️ Télécharger {m}.png",
                    data=buf.getvalue(),
                    file_name=f"{m}.png",
                    mime="image/png",
                )
