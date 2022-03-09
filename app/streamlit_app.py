"""Streamlit demo: type a request, see the table, the generated formula and the computed result.

Run:  streamlit run app/streamlit_app.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from any cwd

from formulaforge.demo import PRESETS, answer, preset_table
from formulaforge.predict import Predictor

RUN_DIR = os.environ.get("FORMULAFORGE_RUN_DIR", "runs/scratch")

st.set_page_config(page_title="FormulaForge", layout="wide")
st.title("FormulaForge")
st.caption("Plain-English request -> Excel formula -> executed on the table")


@st.cache_resource
def load_predictor(path: str):
    return Predictor.load(path)


try:
    predictor = load_predictor(RUN_DIR)
except Exception as exc:  # no trained model yet
    st.error(f"Could not load a trained model from `{RUN_DIR}` ({exc}). Run the training steps in the README first.")
    st.stop()

with st.sidebar:
    source = st.radio("Table", PRESETS + ["upload CSV"])
    seed = st.number_input("Random seed", value=7, step=1)
    beams = st.slider("Beam width", 1, 8, 4)

if source == "upload CSV":
    up = st.file_uploader("CSV file", type="csv")
    df = pd.read_csv(up) if up else preset_table("sales", int(seed))
else:
    df = preset_table(source, int(seed))

shown = df.copy()
shown.index = range(2, len(df) + 2)
shown.columns = [f"{chr(65 + i)}: {c}" for i, c in enumerate(df.columns)]
st.subheader("Table (row 1 is the header)")
st.dataframe(shown, width="stretch")

request = st.text_input("What do you want to compute?", "What is the total Revenue where Region is East?")
if request:
    out = answer(predictor, request, df, beams)
    st.subheader("Generated formula")
    st.code(out["formula"], language="excel")
    st.subheader("Computed result")
    if out["ok"]:
        st.success(str(out["result"]))
    else:
        st.warning("The generated formula could not be evaluated on this table.")
