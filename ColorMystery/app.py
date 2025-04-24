"""
Streamlit UI for ColorMystery.
"""
import json, tempfile
from pathlib import Path
import streamlit as st
from core import stream_process

LOCALES = {}
loc_dir = Path(__file__).with_name('i18n')
for lang in ('en', 'fr'):
    LOCALES[lang] = json.loads((loc_dir / f'{lang}.json').read_text())

def _t(key):
    lang = st.session_state.get('lang', 'en')
    return LOCALES[lang].get(key, key)

st.set_page_config(page_title='ColorMystery', layout='wide')
with st.sidebar:
    st.selectbox('Language / Langue', options=['en', 'fr'], key='lang')
    st.title('ColorMystery')
    mode = st.selectbox(_t('mode'), ['both', 'number', 'mystery'])
    difficulty = st.radio(_t('difficulty'), ['easy', 'medium', 'hard'], index=1)
    detail = st.radio(_t('detail'), ['low', 'medium', 'high'], index=1)
    width = st.number_input(_t('width'), 256, 8192, 1024, 64)
    tile = st.number_input(_t('tile_size'), 256, 2048, 1024, 64)
uploaded = st.file_uploader(_t('upload'), type=['png','jpg','jpeg'])
if uploaded:
    st.image(uploaded, caption=_t('preview'), use_column_width=True)
    if st.button(_t('process')):
        with st.spinner(_t('running')):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(uploaded.read()); tmp.flush()
            class Args: pass
            Args.image = tmp.name
            Args.mode = mode
            Args.difficulty = difficulty
            Args.detail = detail
            Args.id_set = [str(i) for i in range(1,1000)]
            Args.k = {'easy':8,'medium':16,'hard':24}[difficulty]
            Args.width = width
            Args.tile_size = tile
        sheets = stream_process(Path(tmp.name), Args)
        st.success('✓')
        for m, im in sheets.items():
            if mode in (m,'both'):
                st.subheader(m)
                st.image(im, use_column_width=True)
