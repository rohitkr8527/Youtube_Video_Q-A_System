import streamlit as st
from transcript_utils import get_transcript
from chunk_utils import split_transcript_into_chunks
from vector_store_utils import create_vector_store
from qa_utils import build_qa_chain
from export_utils import export_to_json, export_to_pdf

def main():
    col1, col2 = st.columns([1, 10])

    col1.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png",
    width=40,)

    col2.markdown(
    """
    <div style="display: flex; align-items: center; height: 30px; padding-right: 100px; padding-bottom: 10px;">
        <h1 style="margin: 0;">YouTube Q&A System</h1>
    </div>
    """,
    unsafe_allow_html=True,)


    st.markdown("Ask questions about any YouTube video using its transcript.")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        chunk_size = st.slider("Chunk size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200)

    video_url = st.text_input("🔗 Enter YouTube video URL:")

    if video_url:
        # Only reprocess if URL changed
        if "video_url" not in st.session_state or st.session_state.video_url != video_url:
            try:
                with st.spinner("⏳ Processing video..."):
                    full_text, raw_transcript = get_transcript(video_url)
                    st.success("✅ Transcript loaded!")

                    chunks = split_transcript_into_chunks(
                        full_text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                    vectordb = create_vector_store(chunks)
                    qa_chain = build_qa_chain(vectordb)

                    # Save to session state
                    st.session_state.video_url = video_url
                    st.session_state.video_title = "YouTube Q&A"
                    st.session_state.qa_chain = qa_chain
                    st.session_state.qa_pairs = []

                    # Optional: derive simple title from video ID
                    if "v=" in video_url:
                        video_id = video_url.split("v=")[1][:11]
                        st.session_state.video_title = f"Video {video_id} Q&A"

            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                return

        # QA Interface
        st.subheader("🧠 Ask Your Question")
        question = st.text_input("❓ Your Question:")

        if question and "qa_chain" in st.session_state:
            try:
                with st.spinner("🤖 Generating answer..."):
                    response = st.session_state.qa_chain.invoke({"query": question})
                    answer = response["result"]

                    st.session_state.qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })

                    st.text_area("💡 Answer:", value=answer, height=200)
            except Exception as e:
                st.error(f"⚠️ Error generating answer: {str(e)}")

        # Export
        if st.session_state.get("qa_pairs"):
            st.subheader("📁 Export Q&A")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Export to PDF"):
                    export_to_pdf(st.session_state.video_title, video_url, st.session_state.qa_pairs)
                    with open("qa_results.pdf", "rb") as f:
                        st.download_button("Download PDF", f, file_name="qa_results.pdf")

            with col2:
                if st.button("🗂️ Export to JSON"):
                    export_to_json(video_url, st.session_state.video_title, st.session_state.qa_pairs)
                    with open("qa_results.json", "rb") as f:
                        st.download_button("Download JSON", f, file_name="qa_results.json")

if __name__ == "__main__":
    main()

