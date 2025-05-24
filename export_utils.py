import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import inch

def export_to_json(video_url, video_title, qa_pairs, filename="qa_results.json"):
    data = {
        "video_title": video_title,
        "video_url": video_url,
        "qa_pairs": qa_pairs,
        "date_generated": datetime.now().strftime("%Y-%m-%d")
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def export_to_pdf(video_title, video_url, qa_pairs, filename="qa_results.pdf"):
    # Create document with reasonable margins
    doc = SimpleDocTemplate(filename, pagesize=letter,
                          leftMargin=40, rightMargin=40,
                          topMargin=40, bottomMargin=40)
    
    # Create a simple style
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']
    style_normal.fontName = 'Helvetica'
    style_normal.fontSize = 12
    style_normal.leading = 14
    style_normal.alignment = 0  # Left aligned
    
    story = []
    
    # Add video info
    story.append(Paragraph(f"<b>Video:</b> {video_title}", style_normal))
    story.append(Paragraph(f"<b>URL:</b> {video_url}", style_normal))
    story.append(Spacer(1, 0.25*inch))
    
    # Add Q&A pairs
    for idx, qa in enumerate(qa_pairs, 1):
        # Question
        story.append(Paragraph(f"<b>Q{idx}:</b> {qa['question']}", style_normal))
        
        # Answer - split into paragraphs if it contains multiple sentences
        answer = qa['answer']
        story.append(Paragraph(f"<b>A{idx}:</b> {answer}", style_normal))
        story.append(Spacer(1, 0.25*inch))
    
    # Add date
    story.append(Paragraph(f"<b>Date Generated:</b> {datetime.now().strftime('%Y-%m-%d')}", style_normal))
    
    # Build PDF
    doc.build(story)