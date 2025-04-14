import os
import uuid
import json  # <-- Add json import
from flask import Flask, request, jsonify, render_template, abort
from werkzeug.utils import secure_filename
from ollama import chat

# --- Text Extraction Libraries ---
import PyPDF2
import docx  # Python-docx library

# --- Data Handling & Visualization ---
import pandas as pd       # <-- Add pandas
import plotly.express as px  # <-- Available if needed
import plotly.io as pio      # <-- Add Plotly IO for JSON export
import plotly.graph_objects as go  # <-- For complete control
from plotly.subplots import make_subplots

# --- For text processing ---
import re
from collections import Counter

# --- Flask App Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls'}  # <-- Allowed file types

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# --- Ollama Configuration ---
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')

# -------------------------------------------------
#                HELPER FUNCTIONS
# -------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception as decrypt_err:
                    print(f"Could not decrypt PDF {file_path}: {decrypt_err}")
                    return "Error reading PDF: File is encrypted and could not be decrypted."
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text if text else "Could not extract text from PDF (possibly image-based or empty)."
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        if "encrypted" in str(e).lower():
            return "Error reading PDF: File is encrypted."
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        return text if text else "No text found in DOCX document."
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return f"Error reading DOCX: {e}"

def summarize_text_with_ollama(text_content, max_length=30000):
    if not text_content or text_content.isspace():
        return "Cannot summarize empty or whitespace-only text."
    if len(text_content) > max_length:
        print(f"Warning: Text length ({len(text_content)}) exceeds limit ({max_length}). Truncating.")
        text_content = text_content[:max_length] + "\n... [Content Truncated]"
    summarization_prompt = f"""
You are an expert text summarizer. Please provide a concise summary of the following document content.
Focus on the main points, key findings, and overall message. Avoid adding opinions or information not present in the text.
The summary should be easy to understand and capture the essence of the document.

Document Content:
---
{text_content}
---

Concise Summary:
"""
    try:
        response = chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': summarization_prompt}
        ])
        summary = response.get('message', {}).get('content', "").strip()
        return summary if summary else "AI model returned an empty summary."
    except Exception as e:
        print(f"Error calling Ollama for summarization: {e}")
        return f"Error during summarization: {e}"

# -------------------------------------------------
#       NEW: Text to DataFrame Parsing Function
# -------------------------------------------------

def parse_text_to_dataframe(text):
    """
    Attempts to parse the input text into a structured two-column DataFrame.
    It splits the text on newlines and then uses common delimiters (comma, tab, or two+ spaces)
    to split each line into exactly two fields.
    It then tries to convert the second field into a float.
    Returns a DataFrame with columns ["X", "Y"] if successful and at least 2 rows;
    otherwise, returns None.
    """
    lines = text.strip().splitlines()
    data = []
    for line in lines:
        # Try splitting by comma, then by tab, then by two or more spaces.
        if ',' in line:
            parts = line.split(',')
        elif '\t' in line:
            parts = line.split('\t')
        else:
            parts = re.split(r'\s{2,}', line)
        if len(parts) == 2:
            part1 = parts[0].strip()
            part2 = parts[1].strip()
            # Clean part2 (remove $ and commas)
            part2_clean = re.sub(r'[\$,]', '', part2)
            try:
                number = float(part2_clean)
            except Exception:
                # Skip lines where conversion fails.
                continue
            data.append([part1, number])
    if len(data) >= 2:
        df = pd.DataFrame(data, columns=["X", "Y"])
        # Debug: Print the parsed DataFrame
        print("[DEBUG] Parsed DataFrame from text:")
        print(df)
        return df
    # Return None if not enough valid rows
    print("[DEBUG] Failed to parse valid two-column data from text.")
    return None

# -------------------------------------------------
#       TEXT-BASED VISUALIZATION FUNCTIONS
# -------------------------------------------------

def create_bar_chart_from_text(text):
    df = parse_text_to_dataframe(text)
    if df is None or df.empty:
        return None, "The text was not appropriate for visualization (could not extract valid two-column numeric data)."
    x_col = "X"
    y_col = "Y"
    df['unique_id'] = df.index.astype(str) + "_" + df[x_col].astype(str)
    unique_ids = df['unique_id'].tolist()
    categories = df[x_col].tolist()
    values = df[y_col].tolist()
    fig = go.Figure(data=[go.Bar(
        x=unique_ids,
        y=values,
        text=values,
        marker_color='blue',
        customdata=categories,
        hovertemplate=f"{x_col}: %{{customdata}}<br>{y_col}: %{{y}}<br><extra></extra>"
    )])
    fig.update_layout(
        title=f"Visualization of {x_col} vs {y_col} (Bar Chart from Text)",
        xaxis=dict(
            title=x_col,
            tickmode='array',
            tickvals=unique_ids,
            ticktext=categories,
            tickangle=-45 if len(unique_ids) > 5 else 0,
        ),
        yaxis=dict(
            title=y_col,
            range=[0, max(values) * 1.1 if max(values) > 0 else 1],
            gridcolor='LightGray',
            gridwidth=1,
        ),
        bargap=0.2,
        plot_bgcolor='white',
    )
    chart_json_str = pio.to_json(fig)
    chart_json_dict = json.loads(chart_json_str)
    return chart_json_dict, None

def create_pie_chart_from_text(text):
    df = parse_text_to_dataframe(text)
    if df is None or df.empty:
        return None, "The text was not appropriate for visualization (could not extract valid two-column numeric data)."
    x_col = "X"
    y_col = "Y"
    df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
    labels = df_grouped[x_col].tolist()
    values = df_grouped[y_col].tolist()
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        hoverinfo='label+value'
    )])
    fig.update_layout(
        title=f"Visualization of {x_col} vs {y_col} (Pie Chart from Text)"
    )
    chart_json_str = pio.to_json(fig)
    chart_json_dict = json.loads(chart_json_str)
    return chart_json_dict, None

def create_infograph_from_text(text):
    df = parse_text_to_dataframe(text)
    if df is None or df.empty:
        return None, "The text was not appropriate for visualization (could not extract valid two-column numeric data)."
    x_col = "X"
    y_col = "Y"
    df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
    categories = df_grouped[x_col].tolist()
    values = df_grouped[y_col].tolist()
    total_value = sum(values)
    avg_value = total_value / len(values) if values else 0
    max_value = max(values)
    min_value = min(values)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Pie Chart", "Bar Chart", "Donut Chart", "Ring Chart")
    )
    # Top-left: Pie Chart
    fig.add_trace(
        go.Pie(
            labels=categories,
            values=values,
            textinfo='label+percent',
            hoverinfo='label+value'
        ),
        row=1, col=1
    )
    # Top-right: Horizontal Bar Chart
    df_sorted = df_grouped.sort_values(by=y_col, ascending=False)
    fig.add_trace(
        go.Bar(
            x=df_sorted[y_col],
            y=df_sorted[x_col],
            orientation='h',
            text=df_sorted[y_col],
            textposition='auto',
            marker_color='orange'
        ),
        row=1, col=2
    )
    # Bottom-left: Donut Chart
    fig.add_trace(
        go.Pie(
            labels=categories,
            values=values,
            hole=0.4,
            textinfo='percent',
            hoverinfo='label+value'
        ),
        row=2, col=1
    )
    # Bottom-right: Ring Chart
    fig.add_trace(
        go.Pie(
            labels=categories,
            values=values,
            hole=0.7,
            textinfo='none',
            hoverinfo='label+value'
        ),
        row=2, col=2
    )
    fig.update_layout(
        title_text="INFOGRAPHICS (Text Analysis)",
        title_x=0.5,
        title_font_size=24,
        width=800,
        height=1100,
        margin=dict(l=50, r=50, t=100, b=80),
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.05,
            xanchor="center",
            yanchor="top",
            orientation="h"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor='LightGray', row=1, col=2)
    fig.update_yaxes(automargin=True, row=1, col=2)
    chart_json_str = pio.to_json(fig)
    chart_json_dict = json.loads(chart_json_str)
    return chart_json_dict, None

def create_line_graph_from_text(text):
    df = parse_text_to_dataframe(text)
    if df is None or df.empty:
        return None, "The text was not appropriate for visualization (could not extract valid two-column numeric data)."
    x_col = "X"
    y_col = "Y"
    # Attempt to convert the x-column to datetime; if that fails, use row numbers
    try:
        df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        if df[x_col].isnull().all():
            x_values = list(range(1, len(df) + 1))
            x_label = "Row Number"
        else:
            x_values = df[x_col].tolist()
            x_label = x_col
    except Exception:
        x_values = list(range(1, len(df) + 1))
        x_label = "Row Number"
    y_values = df[y_col].tolist()
    fig = go.Figure(data=go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        line=dict(color='green'),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"Line Graph ({x_label} vs {y_col} from Text)",
        xaxis_title=x_label,
        yaxis_title=y_col,
        plot_bgcolor='white'
    )
    chart_json_str = pio.to_json(fig)
    chart_json_dict = json.loads(chart_json_str)
    return chart_json_dict, None

# -------------------------------------------------
#          EXCEL-BASED VISUALIZATION FUNCTIONS
# (Unchanged from previous version)
# -------------------------------------------------

def create_bar_chart_from_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print("[DEBUG] DataFrame loaded (Bar Chart):")
        print(df.head())
        if df.empty:
            return None, "Excel file is empty or has no data in the first sheet."
        if len(df.columns) < 2:
            return None, "Excel file needs at least two columns for visualization (Category, Value)."
        x_col = df.columns[0]
        y_col = df.columns[1]
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            if df[y_col].isnull().all():
                return None, f"The second column ('{y_col}') does not contain numeric data suitable for visualization."
        df['unique_id'] = df.index.astype(str) + "_" + df[x_col].astype(str)
        unique_ids = df['unique_id'].tolist()
        categories = df[x_col].tolist()
        values = df[y_col].tolist()
        fig = go.Figure(data=[go.Bar(
            x=unique_ids,
            y=values,
            text=values,
            marker_color='blue',
            customdata=categories,
            hovertemplate=f"{x_col}: %{{customdata}}<br>{y_col}: %{{y}}<br><extra></extra>"
        )])
        fig.update_layout(
            title=f"Visualization of {x_col} vs {y_col} (Bar Chart)",
            xaxis=dict(
                title=x_col,
                tickmode='array',
                tickvals=unique_ids,
                ticktext=categories,
                tickangle=-45 if len(unique_ids) > 5 else 0,
            ),
            yaxis=dict(
                title=y_col,
                range=[0, max(values) * 1.1 if max(values) > 0 else 1],
                gridcolor='LightGray',
                gridwidth=1,
            ),
            bargap=0.2,
            plot_bgcolor='white',
        )
        chart_json_str = pio.to_json(fig)
        chart_json_dict = json.loads(chart_json_str)
        print("Bar chart JSON generated successfully.")
        return chart_json_dict, None
    except Exception as e:
        print(f"Error in create_bar_chart_from_excel: {e}")
        return None, f"An unexpected error occurred: {e}"

def create_pie_chart_from_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print("[DEBUG] DataFrame loaded (Pie Chart):")
        print(df.head())
        if df.empty:
            return None, "Excel file is empty or has no data in the first sheet."
        if len(df.columns) < 2:
            return None, "Excel file needs at least two columns for visualization (Category, Value)."
        x_col = df.columns[0]
        y_col = df.columns[1]
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            if df[y_col].isnull().all():
                return None, f"The second column ('{y_col}') does not contain numeric data suitable for visualization."
        df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
        labels = df_grouped[x_col].tolist()
        values = df_grouped[y_col].tolist()
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            hoverinfo='label+value'
        )])
        fig.update_layout(
            title=f"Visualization of {x_col} vs {y_col} (Pie Chart)"
        )
        chart_json_str = pio.to_json(fig)
        chart_json_dict = json.loads(chart_json_str)
        print("Pie chart JSON generated successfully.")
        return chart_json_dict, None
    except Exception as e:
        print(f"Error in create_pie_chart_from_excel: {e}")
        return None, f"An unexpected error occurred: {e}"

def create_infograph_from_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print("[DEBUG] DataFrame loaded (Infograph - multi-charts):")
        print(df.head())
        if df.empty:
            return None, "Excel file is empty or has no data in the first sheet."
        if len(df.columns) < 2:
            return None, "Excel file needs at least two columns for visualization (Category, Value)."
        x_col = df.columns[0]
        y_col = df.columns[1]
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            if df[y_col].isnull().all():
                return None, f"The second column ('{y_col}') does not contain numeric data suitable for visualization."
        df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
        categories = df_grouped[x_col].tolist()
        values = df_grouped[y_col].tolist()
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}],
                   [{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=("Pie Chart", "Bar Chart", "Donut Chart", "Ring Chart")
        )
        # Top-left: Pie Chart
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=values,
                textinfo='label+percent',
                hoverinfo='label+value'
            ),
            row=1, col=1
        )
        # Top-right: Horizontal Bar Chart
        df_sorted = df_grouped.sort_values(by=y_col, ascending=False)
        fig.add_trace(
            go.Bar(
                x=df_sorted[y_col],
                y=df_sorted[x_col],
                orientation='h',
                text=df_sorted[y_col],
                textposition='auto',
                marker_color='orange'
            ),
            row=1, col=2
        )
        # Bottom-left: Donut Chart
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=values,
                hole=0.4,
                textinfo='percent',
                hoverinfo='label+value'
            ),
            row=2, col=1
        )
        # Bottom-right: Ring Chart
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=values,
                hole=0.7,
                textinfo='none',
                hoverinfo='label+value'
            ),
            row=2, col=2
        )
        fig.update_layout(
            title_text="INFOGRAPHICS",
            title_x=0.5,
            title_font_size=24,
            width=800,
            height=1100,
            margin=dict(l=50, r=50, t=100, b=80),
            showlegend=True,
            legend=dict(
                x=0.5,
                y=-0.05,
                xanchor="center",
                yanchor="top",
                orientation="h"
            )
        )
        fig.update_xaxes(showgrid=True, gridcolor='LightGray', row=1, col=2)
        fig.update_yaxes(automargin=True, row=1, col=2)
        chart_json_str = pio.to_json(fig)
        chart_json_dict = json.loads(chart_json_str)
        print("Infograph multi-chart JSON (A4 style) generated successfully.")
        return chart_json_dict, None
    except Exception as e:
        print(f"Error in create_infograph_from_excel: {e}")
        return None, f"An unexpected error occurred: {e}"

def create_line_graph_from_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print("[DEBUG] DataFrame loaded (Line Graph):")
        print(df.head())
        if df.empty:
            return None, "Excel file is empty or has no data in the first sheet."
        if len(df.columns) < 2:
            return None, "Excel file needs at least two columns for visualization (Category, Value)."
        y_col = df.columns[1]
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            if df[y_col].isnull().all():
                return None, f"The second column ('{y_col}') does not contain numeric data suitable for visualization."
        x_values = list(range(1, len(df) + 1))
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=df[y_col],
            mode='lines+markers',
            line=dict(color='green'),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f"Line Graph (Row Number vs {y_col})",
            xaxis_title="Row Number",
            yaxis_title=y_col,
            plot_bgcolor='white'
        )
        chart_json_str = pio.to_json(fig)
        chart_json_dict = json.loads(chart_json_str)
        print("Line graph JSON generated successfully.")
        return chart_json_dict, None
    except Exception as e:
        print(f"Error in create_line_graph_from_excel: {e}")
        return None, f"An unexpected error occurred: {e}"

# -------------------------------------------------
#                FLASK ROUTES
# -------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/summarization')
def summarization_page():
    return render_template('summarization.html', active_page='summarization')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/research')
def research():
    return render_template('research.html', active_page='research')

@app.route('/visualization', methods=['GET', 'POST'])
def visualization_page_and_handler():
    if request.method == 'GET':
        return render_template('visualization.html', active_page='visualization')
    
    # If text is provided, process text input.
    if 'text' in request.form and request.form['text'].strip():
        text_input = request.form['text'].strip()
        print("[DEBUG] Processing text visualization.")
        bar_chart_json, err1 = create_bar_chart_from_text(text_input)
        pie_chart_json, err2 = create_pie_chart_from_text(text_input)
        infograph_json, err3 = create_infograph_from_text(text_input)
        line_graph_json, err4 = create_line_graph_from_text(text_input)
        if err1 or err2 or err3 or err4:
            err_msg = err1 or err2 or err3 or err4
            return jsonify({"error": err_msg}), 400
        response_data = {
            "bar_chart": bar_chart_json,
            "pie_chart": pie_chart_json,
            "infograph": infograph_json,
            "line_graph": line_graph_json
        }
        return jsonify(response_data)
    
    # Otherwise, process file upload.
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        if file_extension not in ['xlsx', 'xls']:
            print(f"Incorrect file type for visualization: {original_filename}")
            return jsonify({"error": "File type not allowed for visualization. Please upload XLSX or XLS."}), 400
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(file_path)
            print(f"Excel file saved temporarily to: {file_path}")
            bar_chart_json, err1 = create_bar_chart_from_excel(file_path)
            pie_chart_json, err2 = create_pie_chart_from_excel(file_path)
            infograph_json, err3 = create_infograph_from_excel(file_path)
            line_graph_json, err4 = create_line_graph_from_excel(file_path)
            os.remove(file_path)
            print(f"Temporary file removed: {file_path}")
            if err1 or err2 or err3 or err4:
                err_msg = err1 or err2 or err3 or err4
                print(f"Visualization generation failed: {err_msg}")
                return jsonify({"error": err_msg}), 400
            response_data = {
                "bar_chart": bar_chart_json,
                "pie_chart": pie_chart_json,
                "infograph": infograph_json,
                "line_graph": line_graph_json
            }
            print("Visualization JSON for all charts generated successfully.")
            return jsonify(response_data)
        except Exception as e:
            print(f"An error occurred during visualization processing: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Temporary file removed after error: {file_path}")
                except OSError as remove_err:
                    print(f"Error removing file during exception handling: {remove_err}")
            return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500
    else:
        print(f"File type not allowed or invalid file: {file.filename}")
        return jsonify({"error": "File type not allowed. Please upload XLSX or XLS."}), 400

@app.route('/summarize', methods=['POST'])
def handle_summarize():
    if 'text' in request.form and request.form['text'].strip():
        text_content = request.form['text'].strip()
        if not text_content:
            return jsonify({"error": "Text input is empty. Please provide valid text to summarize."}), 400
        try:
            print(f"Received text input for summarization. Length: {len(text_content)} characters.")
            summary = summarize_text_with_ollama(text_content)
            return jsonify({"summary": summary})
        except Exception as e:
            print(f"An error occurred during text summarization: {e}")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        if file_extension not in ['pdf', 'docx']:
            print(f"Incorrect file type for summarization: {original_filename}")
            return jsonify({"error": "File type not allowed for summarization. Please upload PDF or DOCX."}), 400
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(file_path)
            print(f"File saved temporarily to: {file_path}")
            if file_extension == 'pdf':
                extracted_text = extract_text_from_pdf(file_path)
            else:
                extracted_text = extract_text_from_docx(file_path)
            if extracted_text.startswith("Error reading") or extracted_text.startswith("Could not extract"):
                summary = f"Text Extraction Failed: {extracted_text}"
            elif not extracted_text or extracted_text.isspace():
                summary = "Text extraction resulted in empty content. Cannot summarize."
            else:
                print(f"Extracted text length: {len(extracted_text)}. Summarizing...")
                summary = summarize_text_with_ollama(extracted_text)
            os.remove(file_path)
            print(f"Temporary file removed: {file_path}")
            return jsonify({"summary": summary})
        except Exception as e:
            print(f"An error occurred during summarization processing: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_err:
                    print(f"Error removing file during exception handling: {remove_err}")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    else:
        print(f"File type not allowed (Summarize Endpoint): {file.filename}")
        return jsonify({"error": "File type not allowed. Please upload PDF or DOCX."}), 400

# -------------------------------------------------
#                    RUN THE APP
# -------------------------------------------------
if __name__ == '__main__':
    print(f"Starting Flask Document Summarizer & Visualizer server (PDF, DOCX, XLSX, XLS) using Ollama model '{OLLAMA_MODEL}'...")
    app.run(host='0.0.0.0', port=5001, debug=True)
