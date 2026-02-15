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
    """Ensures text is safe for ReportLab's XML parser and handles Markdown."""
    if not text: return ""
    if isinstance(text, (dict, list)):
        text = json.dumps(text, indent=2)

    text = str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Re-enable specific safe tags
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;br/&gt;', '<br/>').replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')

    # Convert Markdown to PDF-safe HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    for tag in ['b', 'i']:
        open_count = text.count(f'<{tag}>')
        close_count = text.count(f'</{tag}>')
        if open_count > close_count:
            text += f'</{tag}>' * (open_count - close_count)

    return text.replace('\n', '<br/>')

def parse_meal_table(output_text, styles):
    """Resilient parsing for Markdown tables from the Planner Agent."""
    output_text = str(output_text).strip()
    
    if "|" in output_text:
        rows = []
        lines = output_text.split('\n')
        for line in lines:
            if re.match(r'^[\s|:-]+$', line): continue
            if '|' in line:
                raw_cells = [c.strip() for c in line.split('|')]
                if raw_cells and raw_cells[0] == "": raw_cells.pop(0)
                if raw_cells and raw_cells[-1] == "": raw_cells.pop()
                if raw_cells:
                    processed_cells = [Paragraph(sanitize_for_pdf(c), styles['Small']) for c in raw_cells]
                    rows.append(processed_cells)
        
        if len(rows) > 1:
            num_cols = len(rows[0])
            col_widths = [530 / num_cols] * num_cols # Default even distribution
            if num_cols == 6: col_widths = [45, 105, 105, 105, 105, 65]

            try:
                t = Table(rows, colWidths=col_widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E5077")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ]))
                return [t]
            except: pass
            
    return [Paragraph(sanitize_for_pdf(output_text), styles['BodyText'])]

def save_to_pdf(filename, report_payload):
    """
    Saves the final Clinical Report using the ReportAgent's payload.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='Small', fontSize=8, leading=10))
    styles.add(ParagraphStyle(name='CenterHeading', parent=styles['Heading1'], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='VerdictStyle', parent=styles['BodyText'], backColor=colors.whitesmoke, borderPadding=5, borderRadius=3, borderIndex=1))
    
    flowables = []
    metadata = report_payload.get('metadata', {})
    
    # 1. Header & Metrics Dashboard
    flowables.append(Paragraph("Clinical Nutrition Analysis Report", styles['CenterHeading']))
    flowables.append(Paragraph(f"<b>Patient:</b> {metadata.get('patient_id')} | <b>Date:</b> {metadata.get('report_date')}", styles['Small']))
    flowables.append(Spacer(1, 10))
    
    # Dashboard Table
    dash_data = [
        ["BMI", "Kcal Goal", "Protein Goal", "Status"],
        [metadata.get('bmi'), f"{metadata.get('kcal_target')} kcal", f"{metadata.get('protein_target')} g", metadata.get('status')]
    ]
    dash_table = Table(dash_data, colWidths=[100, 100, 100, 130])
    dash_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F0F0F0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    flowables.append(dash_table)
    flowables.append(Spacer(1, 20))

    # 2. Executive Summary
    flowables.append(Paragraph("Executive Summary", styles['Heading2']))
    flowables.append(Paragraph(sanitize_for_pdf(report_payload.get('clinical_summary')), styles['BodyText']))
    flowables.append(Spacer(1, 15))
    
    # 3. Safety Audit & Judge Critique
    flowables.append(Paragraph("Independent Safety Audit", styles['Heading2']))
    verdict_color = colors.green if metadata.get('status') == 'Safe' else colors.orange
    if metadata.get('status') == 'Unsafe': verdict_color = colors.red
    
    flowables.append(Paragraph(f"<b>Verdict:</b> <font color='{verdict_color}'>{metadata.get('status')}</font>", styles['BodyText']))
    flowables.append(Paragraph(sanitize_for_pdf(report_payload.get('judge_critique')), styles['VerdictStyle']))
    flowables.append(Spacer(1, 20))

    # 4. Meal Plan Table
    flowables.append(Paragraph("Proposed 7-Day Dietary Plan", styles['Heading2']))
    flowables.extend(parse_meal_table(report_payload.get('meal_plan_table'), styles))
    flowables.append(PageBreak())

    # 5. Agent Trace (Transparency)
    flowables.append(Paragraph("Reasoning Trace & Decision Log", styles['Heading2']))
    trace = report_payload.get('clinical_trace', [])
    if trace:
        trace_data = [["Agent", "Role", "Summary"]]
        for entry in trace:
            trace_data.append([entry[0], entry[1], Paragraph(sanitize_for_pdf(entry[2]), styles['Small'])])
        
        t_table = Table(trace_data, colWidths=[100, 80, 350])
        t_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEEEEE")),
        ]))
        flowables.append(t_table)

    doc.build(flowables)