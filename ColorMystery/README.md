# ColorMystery

Transform any image into colour‑by‑number or mystery colouring sheets.
Ideal for teachers, parents and hobbyists.

## Quick start (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and drop any picture (≤50 MB).

## Command‑line

```bash
python -m core myphoto.jpg --mode both --difficulty medium
```

## Docker

```bash
docker build -t colormystery .
docker run --rm -p 8501:8501 colormystery
```
