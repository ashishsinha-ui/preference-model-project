"""Streamlit demo: which of two responses would a human prefer, how sure is the model, and is the
decision trustworthy (order check, confidence, length, truncation)?

  streamlit run demo/app.py --server.port 8501 --server.address 127.0.0.1
(on the remote server, forward port 8501 in VS Code's Ports tab and open http://localhost:8501)
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import LOW_CONFIDENCE, MODELS, ORDER_GAP, ROOT, predict  # noqa: E402

st.set_page_config(page_title="Pairwise Preference Model", layout="wide")
st.title("Pairwise LLM Response Preference Predictor")
st.caption("Predicts which response a human would prefer, with calibrated confidence and bias checks. "
           "Both A/B orders are scored and averaged, then temperature-calibrated.")

with st.sidebar:
    model_label = st.selectbox("Model", list(MODELS))
    st.markdown(f"**Flags**\n- needs review if confidence < {LOW_CONFIDENCE:.0%}\n"
                f"- order-sensitive if the two orders disagree by > {ORDER_GAP:.2f}")
    if st.button("Load a random HH test example"):
        import pandas as pd
        df = pd.read_parquet(ROOT / "data/processed/hh_rlhf_hard/test.parquet")
        short = df[(df.len_a < 120) & (df.len_b < 120) & (df.subset == "helpful-base")]
        ex = short.sample(1).iloc[0]
        st.session_state.update(prompt=ex.prompt, resp_a=ex.response_a, resp_b=ex.response_b,
                                example=(ex.prompt, ex.response_a, ex.response_b, "A" if ex.label == 1 else "B"))
    st.caption("Examples come from Anthropic HH-RLHF, which can contain offensive text.")

with st.form("pair"):
    prompt = st.text_area("Prompt / conversation", key="prompt", height=120)
    col_a, col_b = st.columns(2)
    resp_a = col_a.text_area("Response A", key="resp_a", height=220)
    resp_b = col_b.text_area("Response B", key="resp_b", height=220)
    submitted = st.form_submit_button("Predict winner", type="primary")

if submitted:
    if not (prompt.strip() and resp_a.strip() and resp_b.strip()):
        st.error("Please fill in the prompt and both responses.")
        st.stop()
    with st.spinner("Scoring both orders..."):
        r = predict(prompt, resp_a, resp_b, MODELS[model_label])
    b = r["bias_checks"]

    st.subheader(f"Predicted winner: Response {r['winner']}")
    ex = st.session_state.get("example")
    if ex and ex[:3] == (prompt, resp_a, resp_b):  # only while the loaded example is unedited
        st.caption(f"Human annotators preferred Response {ex[3]}.")
    m1, m2, m3 = st.columns(3)
    m1.metric("P(A preferred), calibrated", f"{r['p_a']:.3f}")
    m2.metric("Confidence", f"{r['confidence']:.1%}")
    m3.metric("Order gap |P(A|A,B) - P(A|B,A)|", f"{b['order_gap']:.3f}")
    st.progress(r["p_a"], text=f"Response A  {r['p_a']:.0%}  |  {1 - r['p_a']:.0%}  Response B")

    if r["needs_review"]:
        st.warning("Needs human review: the model is not confident, or its decision depends on the order "
                   "the responses were shown in.")
    else:
        st.success("Confident and order-consistent.")
    if b["order_sensitive"]:
        st.error(f"Order-sensitive: A shown first -> P(A) = {b['p_a_order_ab']:.3f}; "
                 f"A shown second -> P(A) = {b['p_a_order_ba']:.3f}"
                 + (" (the decision flips with the order)." if b["decision_flips_with_order"] else "."))
    if b["winner_is_longer"] and b["length_ratio"] >= 1.5:
        st.info(f"Length note: the winner is {b['length_ratio']:.1f}x longer. In our length test (Test B) "
                "these models picked the longer of two same-content answers 63-68% of the time.")
    trunc = [k for k, v in b["truncated"].items() if v]
    if trunc:
        st.info("Truncated to the token budget (prompt 128, responses 190 tokens): "
                + ", ".join({"prompt": "prompt", "a": "Response A", "b": "Response B"}[k] for k in trunc))
    with st.expander("Details"):
        st.json(r)
