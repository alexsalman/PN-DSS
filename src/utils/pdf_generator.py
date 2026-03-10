# src/utils/pdf_generator.py

import re
import json
import os
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, 
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

def sanitize_for_pdf(text):
    """Global cleaner for PDF text. Strips artifacts and handles formatting."""
    if text is None: return ""
    
    # 1. Convert to string and strip LLM internal monologues
    text = str(text)
    text = re.sub(r'<(think|thought)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'(?i)Thinking\.\.\..*?(\.\.\.done thinking|Let\'s think).*?\.?', '', text, flags=re.DOTALL)
    
    # 2. Aggressive artifact removal (Non-ASCII "Nuclear Option")
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = text.replace('<br>', '\n').replace('<br/>', '\n')
    
    # 3. Handle XML Escaping while preserving intended bold/italic tags
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    
    # Convert Markdown ** to PDF <b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Clean up double spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip().replace('\n', '<br/>')

def parse_markdown_table_to_list(text, remove_repeated_headers=True):
    """Resilient MD table parser."""
    rows = []
    text_str = str(text)
    start_idx = text_str.find('|')
    if start_idx == -1: return []
    
    lines = text_str[start_idx:].strip().split('\n')
    header_keywords = ["Day", "Breakfast", "Lunch", "Dinner", "Metric", "Target"]
    
    for line in lines:
        if re.match(r'^[\s|:-]+$', line): continue 
        if '|' in line:
            cells = [c.strip() for c in line.split('|')]
            if cells and cells[0] == "": cells.pop(0)
            if cells and cells[-1] == "": cells.pop()
            
            if remove_repeated_headers and len(rows) > 0:
                if any(k in cells for k in header_keywords) and len(cells) < 3:
                    continue
            
            if cells: rows.append(cells)
    return rows

def build_reportlab_table(data, col_widths, styles, is_header_blue=True):
    """Constructs tables with internal padding and sanitized cell content."""
    processed_rows = [[Paragraph(sanitize_for_pdf(cell), styles['Small']) for cell in row] for row in data]
    t = Table(processed_rows, colWidths=col_widths, repeatRows=1)
    
    bg = colors.HexColor("#2E5077") if is_header_blue else colors.HexColor("#D1D1D1")
    txt = colors.whitesmoke if is_header_blue else colors.black
    
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), bg),
        ('TEXTCOLOR', (0,0), (-1,0), txt),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    return t

def save_to_pdf(filename, report_payload):
    """Generates the PN-DSS Clinical Report with cleaned formatting and latency tracking."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Small', fontSize=8, leading=10))
    styles.add(ParagraphStyle(name='CenterHeading', parent=styles['Heading1'], alignment=TA_CENTER))
    
    flowables = []
    meta = report_payload.get('metadata', {})
    total_time = meta.get('total_generation_time', 'N/A')

    # 1. HEADER
    flowables.append(Paragraph(sanitize_for_pdf(meta.get('report_title', "PN-DSS Report")), styles['CenterHeading']))
    flowables.append(Paragraph(
        f"<b>Patient ID:</b> {meta.get('patient_id')} | "
        f"<b>Generated:</b> {meta.get('report_date')} | "
        f"<b>Latency:</b> {total_time}s", 
        styles['Small']
    ))
    flowables.append(Spacer(1, 20))

    # 2. MEAL PLAN
    flowables.append(Paragraph("Proposed Dietary Plan", styles['Heading2']))
    meal_data = parse_markdown_table_to_list(report_payload.get('meal_plan_table', ''))
    if meal_data:
        # Check if header needs injection
        if not any("Day" in str(cell) for cell in meal_data[0]):
            meal_data.insert(0, ["Day", "Breakfast", "Lunch", "Dinner"])
        flowables.append(build_reportlab_table(meal_data, [40, 160, 160, 160], styles, True))
    
    flowables.append(Spacer(1, 20))

    # 3. EVALUATION & AUDIT
    flowables.append(Paragraph("Clinical Evaluation & Audit", styles['Heading1']))
    flowables.append(Spacer(1, 10))

    # Layer 1: Technical Audit
    flowables.append(Paragraph("Layer 1: Technical Audit", styles['Heading2']))
    l1 = report_payload.get('layer_1_audit', {})
    
    trace = l1.get('agent_trace', [])
    if trace:
        trace_data = [["Specialist Agent", "Clinical Rationale", "Time"]]
        for e in trace:
            # Handle float or N/A duration
            dur_val = e.get('duration', 'N/A')
            dur_str = f"{dur_val}s" if isinstance(dur_val, (int, float)) else f"{dur_val}s"
            
            trace_data.append([
                str(e.get('agent', 'Unknown')), 
                str(e.get('logic', 'N/A')), 
                dur_str
            ])
            
        flowables.append(build_reportlab_table(trace_data, [90, 380, 60], styles, False))
    
    flowables.append(Spacer(1, 15))

    # Verdict Visualization
    status = str(l1.get('verdict', 'CAUTION')).upper()
    v_color = colors.darkgreen if status == 'SAFE' else colors.red if status == 'UNSAFE' else colors.darkorange
    flowables.append(Paragraph(f"<b>Overall Audit Verdict:</b> <font color='{v_color}'>{status}</font>", styles['BodyText']))
    
    audit_data = parse_markdown_table_to_list(l1.get('structured_comparison', ''))
    if audit_data:
        flowables.append(Spacer(1, 5))
        flowables.append(build_reportlab_table(audit_data, [70, 70, 70, 50, 270], styles, False))

    flowables.append(Spacer(1, 20))

    # Layer 2: DQI-I Analysis
    flowables.append(Paragraph("Layer 2: Quantitative Analysis (DQI-I)", styles['Heading2']))
    l2 = report_payload.get('layer_2_quantitative', {})
    
    raw_score_str = str(l2.get('total_score', '0')).split('/')[0]
    total_score = re.sub(r'[^0-9.]', '', raw_score_str)
    
    dqi_table = [
        ["Category", "Score", "Max Points"],
        ["Variety", str(l2.get('variety_score', 0)), "20"],
        ["Adequacy", str(l2.get('adequacy_score', 0)), "40"],
        ["Moderation", str(l2.get('moderation_score', 0)), "30"],
        ["Balance", str(l2.get('balance_score', 0)), "10"],
        ["Total Index", f"<b>{total_score}</b>", "100"]
    ]
    flowables.append(build_reportlab_table(dqi_table, [180, 175, 175], styles, False))

    doc.build(flowables)
