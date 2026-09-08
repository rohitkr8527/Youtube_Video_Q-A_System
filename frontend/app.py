from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from frontend.api_client import APIError, ask, notes, process_video, quiz, summary
from frontend.components import brand, source_chips, video_header
from frontend.state import init_state, reset_video_state
from frontend.styles import CSS


st.set_page_config(
    page_title="VideoRAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)
init_state()


def landing() -> None:
    brand()
    st.markdown(
        """
        <div class="vr-hero">
            <h1>Understand any YouTube video.</h1>
            <p>Ask questions, find important moments, create notes, summaries and quizzes directly from the video.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    center_left, center, center_right = st.columns([1.6, 5, 1.6])
    with center:
        with st.form("video_form", border=False):
            url = st.text_input(
                "YouTube URL",
                placeholder="Paste a YouTube link",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Analyze video", type="primary", use_container_width=True)
        if submitted:
            if not url.strip():
                st.warning("Paste a YouTube URL first.")
                return
            try:
                with st.spinner("Preparing your video..."):
                    st.session_state.video = process_video(url.strip())
                    st.session_state.messages = []
                    st.session_state.summary = None
                    st.session_state.notes = None
                    st.session_state.quiz = None
                st.rerun()
            except APIError as exc:
                st.error(str(exc))


def ask_mode(video: dict) -> None:
    st.markdown('<div class="vr-section-title">Ask about this video</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown('<div class="vr-subtle">Try one of these to get started.</div>', unsafe_allow_html=True)
        suggestions = [
            "What is this video mainly about?",
            "What are the key ideas?",
            "Explain the hardest concept simply.",
            "What should I remember from this video?",
        ]
        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"suggestion-{idx}", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                source_chips(message.get("sources", []))

    question = st.chat_input("Ask anything about this video")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[-8:]
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Finding the relevant part..."):
                    response = ask(video["video_id"], question, history)
                st.markdown(response["answer"])
                source_chips(response.get("sources", []))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                }
            )
        except APIError as exc:
            st.error(str(exc))


def summary_mode(video: dict) -> None:
    st.markdown('<div class="vr-section-title">Video summary</div>', unsafe_allow_html=True)
    if st.session_state.summary is None:
        if st.button("Create summary", type="primary"):
            try:
                with st.spinner("Creating a clear summary..."):
                    st.session_state.summary = summary(video["video_id"])
                st.rerun()
            except APIError as exc:
                st.error(str(exc))
        else:
            st.markdown('<div class="vr-subtle">Get the main ideas and takeaways without watching the full video.</div>', unsafe_allow_html=True)
            return
    st.markdown(st.session_state.summary["content"])
    source_chips(st.session_state.summary.get("sources", []))


def notes_mode(video: dict) -> None:
    st.markdown('<div class="vr-section-title">Study notes</div>', unsafe_allow_html=True)
    if st.session_state.notes is None:
        if st.button("Create notes", type="primary"):
            try:
                with st.spinner("Organizing the important ideas..."):
                    st.session_state.notes = notes(video["video_id"])
                st.rerun()
            except APIError as exc:
                st.error(str(exc))
        else:
            st.markdown('<div class="vr-subtle">Turn the video into clean, structured notes you can review later.</div>', unsafe_allow_html=True)
            return
    st.markdown(st.session_state.notes["content"])
    source_chips(st.session_state.notes.get("sources", []))


def quiz_mode(video: dict) -> None:
    st.markdown('<div class="vr-section-title">Quiz yourself</div>', unsafe_allow_html=True)
    if st.session_state.quiz is None:
        if st.button("Start quiz", type="primary"):
            try:
                with st.spinner("Creating your quiz..."):
                    st.session_state.quiz = quiz(video["video_id"])
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answered = False
                    st.session_state.quiz_selected = None
                st.rerun()
            except APIError as exc:
                st.error(str(exc))
        else:
            st.markdown('<div class="vr-subtle">Test what you understood from the video with one question at a time.</div>', unsafe_allow_html=True)
        return

    questions = st.session_state.quiz.get("questions", [])
    if not questions:
        st.info("A quiz could not be created from this video.")
        return

    index = st.session_state.quiz_index
    if index >= len(questions):
        st.success(f"Quiz complete — {st.session_state.quiz_score}/{len(questions)} correct")
        if st.button("Take it again"):
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.session_state.quiz_selected = None
            st.rerun()
        return

    item = questions[index]
    st.markdown(f'<div class="vr-quiz-counter">Question {index + 1} of {len(questions)}</div>', unsafe_allow_html=True)
    st.progress(index / len(questions))
    st.markdown(f'<div class="vr-quiz-question">{item["question"]}</div>', unsafe_allow_html=True)

    selected = st.radio(
        "Choose an answer",
        options=list(range(len(item["options"]))),
        format_func=lambda i: item["options"][i],
        index=st.session_state.quiz_selected,
        key=f"quiz-radio-{index}",
        label_visibility="collapsed",
        disabled=st.session_state.quiz_answered,
    )
    st.session_state.quiz_selected = selected

    if not st.session_state.quiz_answered:
        if st.button("Check answer", type="primary", disabled=selected is None):
            st.session_state.quiz_answered = True
            if selected == item["correct_index"]:
                st.session_state.quiz_score += 1
            st.rerun()
    else:
        if selected == item["correct_index"]:
            st.success("Correct")
        else:
            st.error(f'Correct answer: {item["options"][item["correct_index"]]}')
        st.markdown(item["explanation"])
        if item.get("source"):
            source_chips([item["source"]])
        if st.button("Next question", type="primary"):
            st.session_state.quiz_index += 1
            st.session_state.quiz_answered = False
            st.session_state.quiz_selected = None
            st.rerun()


def workspace() -> None:
    video = st.session_state.video
    top_left, top_right = st.columns([8, 2])
    with top_left:
        brand()
    with top_right:
        if st.button("New video", use_container_width=True):
            reset_video_state()
            st.rerun()

    st.markdown('<div class="vr-divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.22, 1], gap="large")
    with left:
        st.video(video["url"])
        video_header(video)

    with right:
        mode = st.radio(
            "Mode",
            ["Ask", "Summary", "Notes", "Quiz"],
            horizontal=True,
            label_visibility="collapsed",
            key="mode",
        )
        st.markdown('<div class="vr-divider"></div>', unsafe_allow_html=True)
        with st.container(height=500, border=False):
            if mode == "Ask":
                ask_mode(video)
            elif mode == "Summary":
                summary_mode(video)
            elif mode == "Notes":
                notes_mode(video)
            else:
                quiz_mode(video)


if st.session_state.video is None:
    landing()
else:
    workspace()
