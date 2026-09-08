from __future__ import annotations

import html

import streamlit as st


def brand() -> None:
    st.markdown(
        '<div class="vr-brand"><span class="vr-logo"></span><span>VideoRAG</span></div>',
        unsafe_allow_html=True,
    )


def source_chips(sources: list[dict]) -> None:
    if not sources:
        return
    links = []
    for source in sources:
        label = html.escape(source.get("label", source.get("timestamp", "Source")))
        url = html.escape(source.get("url", ""), quote=True)
        links.append(f'<a class="vr-source" href="{url}" target="_blank">{label}</a>')
    st.markdown('<div class="vr-source-title">From the video</div>', unsafe_allow_html=True)
    st.markdown('<div class="vr-source-row">' + "".join(links) + "</div>", unsafe_allow_html=True)


def video_header(video: dict) -> None:
    title = html.escape(video.get("title", "YouTube video"))
    st.markdown(f'<div class="vr-video-title">{title}</div>', unsafe_allow_html=True)
    duration = video.get("duration_seconds")
    if duration:
        minutes = int(duration // 60)
        st.markdown(f'<div class="vr-subtle">{minutes} min video</div>', unsafe_allow_html=True)
