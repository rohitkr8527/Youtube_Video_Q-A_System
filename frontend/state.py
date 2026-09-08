from __future__ import annotations

import streamlit as st


def init_state() -> None:
    defaults = {
        "video": None,
        "messages": [],
        "mode": "Ask",
        "summary": None,
        "notes": None,
        "quiz": None,
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_answered": False,
        "quiz_selected": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_video_state() -> None:
    for key in ["video", "messages", "summary", "notes", "quiz"]:
        st.session_state[key] = [] if key == "messages" else None
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False
    st.session_state.quiz_selected = None
    st.session_state.mode = "Ask"
