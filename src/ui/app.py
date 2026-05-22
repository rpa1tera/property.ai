import os
import sys
from pathlib import Path

# Garante que o root do projeto está no path quando rodado via `streamlit run`
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.chatbot.chain import answer
from src.rag.embeddings import load_vectorstore

st.set_page_config(
    page_title="property.ai",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS mínimo para identidade visual
st.markdown(
    """
    <style>
    .property-header { font-size: 2rem; font-weight: 700; color: #1a3a5c; }
    .property-sub { color: #5a7a9a; font-size: 0.95rem; margin-bottom: 1rem; }
    .source-tag { background: #e8f0fe; border-radius: 4px; padding: 2px 6px;
                  font-size: 0.8rem; color: #1a3a5c; margin-right: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="property-header">property.ai</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="property-sub">ChatBot especializado em Seguros Patrimoniais (Property) · InsurMinds</div>',
    unsafe_allow_html=True,
)
st.divider()


@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def _load_vs():
    return load_vectorstore()


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reformulation_count" not in st.session_state:
        st.session_state.reformulation_count = 0


_init_state()

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            cols = st.columns([1])
            with st.expander("Fontes consultadas", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)
        if msg.get("escalated"):
            st.info("Esta consulta foi encaminhada para um especialista humano.")

# Input
if prompt := st.chat_input("Digite sua pergunta sobre seguros patrimoniais..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando na base de conhecimento..."):
            try:
                vs = _load_vs()
                # UI usa caminho rápido: sem RAG-Fusion (LLM extra), sem CRAG (LLM extra),
                # com streaming. RAGAS continua usando defaults (fusion+crag) para baseline.
                result = answer(
                    prompt,
                    vectorstore=vs,
                    stream=True,
                    enable_fusion=False,
                    enable_crag=False,
                )
            except Exception as exc:
                result = {
                    "stream": None,
                    "answer": f"Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente.\n\n_{exc}_",
                    "sources": [],
                    "intent": "erro",
                    "escalated": False,
                }

        if result["stream"] is not None:
            try:
                placeholder = st.empty()
                full_text = ""
                for token in result["stream"]:
                    full_text += token
                    placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                result["answer"] = full_text
            except Exception as exc:
                result["answer"] = f"Erro ao gerar resposta: {exc}"
                st.markdown(result["answer"])
        else:
            st.write(result["answer"])

        if result["sources"]:
            with st.expander("Fontes consultadas", expanded=False):
                for src in result["sources"]:
                    st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

        if result["escalated"]:
            st.info("Esta consulta foi encaminhada para um especialista humano.")

        # Reformulação só se intent=outros E sem resposta útil (escalado)
        if result["intent"] == "outros" and result["escalated"]:
            st.session_state.reformulation_count += 1
            if st.session_state.reformulation_count < 2:
                st.warning("Não entendi bem sua pergunta. Poderia reformulá-la com mais detalhes?")
        else:
            st.session_state.reformulation_count = 0

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "escalated": result["escalated"],
        "intent": result["intent"],
    })
