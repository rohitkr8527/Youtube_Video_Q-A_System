from transcript_utils import get_transcript
from chunk_utils import split_transcript_into_chunks
from vector_store_utils import create_vector_store
from qa_utils import build_qa_chain
from export_utils import export_to_json, export_to_pdf

# Step 1: Get video URL from user
video_url = input("🔗 Enter YouTube video URL: ").strip()

# Step 2: Fetch transcript
try:
    full_text, raw_transcript = get_transcript(video_url)
except Exception as e:
    print(f"❌ Failed to get transcript: {e}")
    exit(1)

# Optional: derive a simple title from URL or use default
video_title = "YouTube Q&A"

# Step 3: Split transcript into chunks
chunks = split_transcript_into_chunks(full_text)

# Step 4: Create Chroma vector store
vectordb = create_vector_store(chunks)

# Step 5: Build QA chain
qa_chain = build_qa_chain(vectordb)

# Step 6: Interactive Q&A loop
qa_pairs = []
print("\n🧠 Ask your questions about the video.")
print("📄 Type 'save' to export Q&A, or 'exit' to quit without saving.\n")

while True:
    user_input = input("❓ You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Exiting without saving.")
        break

    elif user_input.lower() == "save":
        export_to_json(video_url, video_title, qa_pairs)
        export_to_pdf(video_title, video_url, qa_pairs)
        print("\n✅ Q&A saved as JSON and PDF.")
        break

    else:
        try:
            response = qa_chain.invoke({"query": user_input})
            answer = response["result"]
            print(f"🤖 Answer: {answer}\n")

            qa_pairs.append({
                "question": user_input,
                "answer": answer
            })
        except Exception as e:
            print(f"⚠️ Error during answer generation: {e}")
