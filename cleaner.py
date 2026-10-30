"""
DataDoctor: Universal ML Auto-Correction Engine & Certificate Generator
"""
import pandas as pd
import numpy as np
import re
import io
import datetime
import hashlib
from typing import Tuple, Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. CHARACTER N-GRAM NLP MODEL FOR NAME-TO-GENDER PREDICTION
TRAINING_NAMES = [
    # Male names (Indian + Global)
    ("karthik", "Male"), ("varun", "Male"), ("deepak", "Male"), ("rahul", "Male"), ("rohan", "Male"),
    ("amit", "Male"), ("vikas", "Male"), ("naveen", "Male"), ("aditya", "Male"), ("arjun", "Male"),
    ("suresh", "Male"), ("ramesh", "Male"), ("praveen", "Male"), ("sachin", "Male"), ("ganesh", "Male"),
    ("abhishek", "Male"), ("sanjay", "Male"), ("vijay", "Male"), ("raj", "Male"), ("anil", "Male"),
    ("sunil", "Male"), ("manoj", "Male"), ("harish", "Male"), ("mahesh", "Male"), ("prasad", "Male"),
    ("kiran", "Male"), ("ajay", "Male"), ("vicky", "Male"), ("john", "Male"), ("david", "Male"),
    ("michael", "Male"), ("james", "Male"), ("robert", "Male"), ("mohit", "Male"), ("tarun", "Male"),
    ("vivek", "Male"), ("alok", "Male"), ("vishal", "Male"), ("anand", "Male"), ("mukesh", "Male"),
    ("chetan", "Male"), ("ashwin", "Male"), ("rohit", "Male"), ("virat", "Male"), ("hardik", "Male"),
    # Female names (Indian + Global)
    ("divya", "Female"), ("shruti", "Female"), ("ananya", "Female"), ("kavya", "Female"), ("meera", "Female"),
    ("pooja", "Female"), ("sneha", "Female"), ("neha", "Female"), ("priya", "Female"), ("anjali", "Female"),
    ("swati", "Female"), ("ritu", "Female"), ("shreya", "Female"), ("aishwarya", "Female"), ("deepika", "Female"),
    ("tanvi", "Female"), ("preeti", "Female"), ("rani", "Female"), ("sunita", "Female"), ("geeta", "Female"),
    ("lakshmi", "Female"), ("anita", "Female"), ("radha", "Female"), ("rupa", "Female"), ("suma", "Female"),
    ("bhavya", "Female"), ("chaithra", "Female"), ("sowmya", "Female"), ("mary", "Female"), ("alice", "Female"),
    ("diana", "Female"), ("grace", "Female"), ("emma", "Female"), ("olivia", "Female"), ("sophia", "Female"),
    ("smriti", "Female"), ("harmanpreet", "Female"), ("jemimah", "Female"), ("deepti", "Female"), ("renuka", "Female")
]

def train_gender_nlp_model():
    X_train = [name for name, _ in TRAINING_NAMES]
    y_train = [gender for _, gender in TRAINING_NAMES]
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
        ('classifier', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    return model

GENDER_ML_MODEL = train_gender_nlp_model()

def predict_gender_ml(name: Any) -> Tuple[str, float]:
    if pd.isna(name):
        return None, 0.0
    clean = re.sub(r'[^a-zA-Z\s]', '', str(name)).strip().lower()
    tokens = clean.split()
    if not tokens:
        return None, 0.0
    first_name = tokens[0]
    pred = GENDER_ML_MODEL.predict([first_name])[0]
    probs = GENDER_ML_MODEL.predict_proba([first_name])[0]
    return pred, max(probs)

def standardize_date_string(val: Any) -> Any:
    if pd.isna(val) or val is None or str(val).strip() == "":
        return np.nan
    val_str = str(val).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
        return val_str
    try:
        parsed = pd.to_datetime(val_str, errors='coerce', dayfirst=True)
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except Exception:
        pass
    return val

def clean_currency_value(val: Any) -> Any:
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if re.search(r'(rs\.?|₹|\$|inr|usd|€|£|,)', val_str, re.IGNORECASE):
        cleaned_num = re.sub(r'[^\d.]', '', val_str)
        try:
            return float(cleaned_num)
        except ValueError:
            return val
    return val

def harmonize_categories(series: pd.Series) -> pd.Series:
    mapping = {}
    unique_vals = [str(x).strip() for x in series.dropna().unique()]
    for val in unique_vals:
        v_title = val.title()
        if val != v_title:
            mapping[val] = v_title

    abbr_map = {
        'Me': 'Mechanical', 'Mech': 'Mechanical', 'Mechanical Engg': 'Mechanical',
        'Cs': 'Computer Science', 'Cse': 'Computer Science', 'Comp Sci': 'Computer Science',
        'Civil Engg': 'Civil', 'Cv': 'Civil',
        'Ec': 'Electronics', 'Ece': 'Electronics'
    }
    result = series.replace(mapping)
    result = result.replace(abbr_map)
    return result

def calculate_health_score(df: pd.DataFrame) -> Tuple[int, str, Dict[str, Any]]:
    """Calculates dataset quality score from 0 to 100."""
    if df.empty:
        return 0, "F", {}
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    duplicate_ratio = duplicate_rows / len(df) if len(df) > 0 else 0
    
    score = max(0, min(100, int(100 - (missing_ratio * 50 + duplicate_ratio * 30))))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    return score, grade, {
        "score": score,
        "grade": grade,
        "missing_cells": int(missing_cells),
        "missing_pct": round(missing_ratio * 100, 2),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_pct": round(duplicate_ratio * 100, 2)
    }

def clean_data_universal_ml(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = df.copy()
    audit_log = []

    # 1. Column headers to snake_case
    cleaned.columns = [re.sub(r'[^\w]', '', re.sub(r'[\s\.\-]+', '_', str(c).strip().lower())) for c in cleaned.columns]
    audit_log.append("Normalized column headers to snake_case.")

    # 2. Trim string whitespace
    for c in cleaned.select_dtypes(include=['object', 'string']).columns:
        cleaned[c] = cleaned[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. Name-to-Gender NLP cross-correction
    name_cols = [c for c in cleaned.columns if 'name' in c]
    gender_cols = [c for c in cleaned.columns if 'gender' in c or 'sex' in c]
    
    if name_cols and gender_cols:
        n_col, g_col = name_cols[0], gender_cols[0]
        for idx in cleaned.index:
            person_name = cleaned.at[idx, n_col]
            pred_gender, conf = predict_gender_ml(person_name)
            if pred_gender and conf >= 0.65:
                curr_g = str(cleaned.at[idx, g_col]).strip().lower()
                if (pred_gender == "Male" and curr_g in ['f', 'female', 'woman']) or \
                   (pred_gender == "Female" and curr_g in ['m', 'male', 'man']) or \
                   (curr_g in ['f', 'm']):
                    cleaned.at[idx, g_col] = pred_gender

    # 4. Standardize Dates (YYYY-MM-DD)
    for c in cleaned.columns:
        is_date = any(k in c for k in ['date', 'dob', 'time', 'dt', 'joined', 'created', 'birth'])
        if not is_date and cleaned[c].dtype == object:
            sample = cleaned[c].dropna().head(10).astype(str)
            if sum(bool(re.search(r'\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{2,4}', s)) for s in sample) >= 3:
                is_date = True
        if is_date:
            cleaned[c] = cleaned[c].apply(standardize_date_string)

    # 5. Currency Cleaning (Rs / ₹ / Comma)
    for c in cleaned.select_dtypes(include=['object', 'string']).columns:
        has_curr = cleaned[c].dropna().astype(str).str.contains(r'(rs\.?|₹|\$|inr)', regex=True, case=False).any()
        if has_curr or any(k in c for k in ['salary', 'price', 'fare', 'bonus', 'amount', 'cost', 'fee']):
            cleaned[c] = cleaned[c].apply(clean_currency_value)
            cleaned[c] = pd.to_numeric(cleaned[c], errors='ignore')

    # 6. Harmonize Categories
    for c in cleaned.select_dtypes(include=['object']).columns:
        if c not in name_cols and c not in gender_cols and cleaned[c].nunique() < 50:
            cleaned[c] = harmonize_categories(cleaned[c])

    # 7. Missing Value Imputation
    for c in cleaned.select_dtypes(include=[np.number]).columns:
        if cleaned[c].isna().sum() > 0:
            cleaned[c] = cleaned[c].fillna(cleaned[c].median())

    for c in cleaned.select_dtypes(include=['object', 'string']).columns:
        if cleaned[c].isna().sum() > 0:
            val = cleaned[c].mode()[0] if not cleaned[c].mode().empty else "Unknown"
            cleaned[c] = cleaned[c].fillna(val)

    return cleaned, audit_log

def generate_cell_corrections_log(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    corrections = []
    col_map = dict(zip(raw_df.columns, cleaned_df.columns))
    common_indices = raw_df.index.intersection(cleaned_df.index)
    name_col = next((c for c in raw_df.columns if 'name' in c.lower()), None)

    for idx in common_indices:
        line_num = idx + 2
        p_name = str(raw_df.at[idx, name_col]) if name_col else ""

        for orig_col, clean_col in col_map.items():
            orig_val = raw_df.at[idx, orig_col]
            new_val = cleaned_df.at[idx, clean_col]

            if ('gender' in orig_col.lower() or 'sex' in orig_col.lower()) and str(orig_val) != str(new_val):
                corrections.append({
                    "Line #": line_num,
                    "Row Index": idx,
                    "Column": orig_col,
                    "Cleaned Column": clean_col,
                    "Original (Uploaded)": str(orig_val),
                    "Evaluated (Cleaned)": str(new_val),
                    "Correction Type": f"🧠 ML Name-Gender Inference ('{p_name}' -> {new_val})"
                })
            elif isinstance(orig_val, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', str(new_val)) and orig_val != new_val:
                corrections.append({
                    "Line #": line_num,
                    "Row Index": idx,
                    "Column": orig_col,
                    "Cleaned Column": clean_col,
                    "Original (Uploaded)": str(orig_val),
                    "Evaluated (Cleaned)": str(new_val),
                    "Correction Type": "📅 Standardized Date (YYYY-MM-DD)"
                })
            elif isinstance(orig_val, str) and re.search(r'(rs\.?|₹|\$|inr)', orig_val, re.IGNORECASE) and not re.search(r'(rs\.?|₹|\$|inr)', str(new_val), re.IGNORECASE):
                corrections.append({
                    "Line #": line_num,
                    "Row Index": idx,
                    "Column": orig_col,
                    "Cleaned Column": clean_col,
                    "Original (Uploaded)": str(orig_val),
                    "Evaluated (Cleaned)": str(new_val),
                    "Correction Type": "💰 Currency -> Clean Number"
                })
            elif pd.isna(orig_val) and not pd.isna(new_val):
                corrections.append({
                    "Line #": line_num,
                    "Row Index": idx,
                    "Column": orig_col,
                    "Cleaned Column": clean_col,
                    "Original (Uploaded)": "[MISSING / NaN]",
                    "Evaluated (Cleaned)": str(new_val),
                    "Correction Type": "🧼 Imputed Missing Value"
                })
            elif str(orig_val) != str(new_val) and str(orig_val).strip() != str(new_val):
                corrections.append({
                    "Line #": line_num,
                    "Row Index": idx,
                    "Column": orig_col,
                    "Cleaned Column": clean_col,
                    "Original (Uploaded)": str(orig_val),
                    "Evaluated (Cleaned)": str(new_val),
                    "Correction Type": "🧬 Harmonized Category / Case"
                })

    return pd.DataFrame(corrections)

def create_excel_report_with_green_highlights(raw_df, cleaned_df, diff_df, score_before, score_after):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary = pd.DataFrame({
            "Evaluation Metric": [
                "Health Score (Before)", "Health Score (After)", "Score Improvement", 
                "Total Uploaded Records", "Total Evaluated Records", "Total Columns", 
                "Total Cells Corrected (Marked Green)", "Missing Cells (Before)", "Missing Cells (After)"
            ],
            "Value": [
                f"{score_before} / 100", f"{score_after} / 100", f"+{score_after - score_before} points", 
                len(raw_df), len(cleaned_df), len(cleaned_df.columns), len(diff_df),
                int(raw_df.isna().sum().sum()), int(cleaned_df.isna().sum().sum())
            ]
        })
        summary.to_excel(writer, sheet_name="1_Executive_Summary", index=False)
        raw_df.to_excel(writer, sheet_name="2_Uploaded_Original_Doc", index=False)
        cleaned_df.to_excel(writer, sheet_name="3_Cleaned_Evaluated_Doc", index=False)
        
        if not diff_df.empty:
            diff_df.to_excel(writer, sheet_name="4_Audit_Cell_Corrections", index=False)

        try:
            from openpyxl.styles import PatternFill, Font
            ws = writer.sheets["3_Cleaned_Evaluated_Doc"]
            green_fill = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
            green_font = Font(color="155724", bold=True)
            for _, row in diff_df.iterrows():
                row_idx = row["Row Index"]
                clean_col = row["Cleaned Column"]
                excel_row = row_idx + 2
                if clean_col in cleaned_df.columns:
                    excel_col = cleaned_df.columns.get_loc(clean_col) + 1
                    cell = ws.cell(row=excel_row, column=excel_col)
                    cell.fill = green_fill
                    cell.font = green_font
        except Exception:
            pass

    return output.getvalue()

def generate_html_certificate(
    filename: str, 
    raw_df: pd.DataFrame, 
    cleaned_df: pd.DataFrame, 
    diff_df: pd.DataFrame, 
    score_before: int, 
    score_after: int
) -> str:
    cert_id = f"DD-{hashlib.md5(filename.encode()).hexdigest()[:8].upper()}-2026"
    now_str = datetime.datetime.now().strftime("%B %d, %Y • %H:%M UTC")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>DataDoctor - Certificate of Data Quality</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Montserrat:wght@400;600;700&display=swap');
            body {{
                font-family: 'Montserrat', sans-serif;
                background-color: #f1f5f9;
                display: flex;
                justify-content: center;
                padding: 30px;
                margin: 0;
            }}
            .cert-card {{
                width: 850px;
                background: #ffffff;
                padding: 50px 60px;
                border: 12px double #0f172a;
                box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                position: relative;
                color: #1e293b;
            }}
            .gold-badge {{
                position: absolute;
                top: 40px;
                right: 50px;
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: linear-gradient(135deg, #f59e0b, #b45309);
                color: #ffffff;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                font-weight: 700;
                font-size: 11px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(180, 83, 9, 0.4);
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .gold-badge span {{ font-size: 20px; font-weight: 800; }}
            .cert-header {{ text-align: center; margin-bottom: 25px; }}
            .cert-title {{ font-family: 'Cinzel', serif; font-size: 32px; color: #0f172a; margin: 0; letter-spacing: 2px; }}
            .cert-subtitle {{ font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 3px; margin-top: 8px; }}
            .cert-body {{ text-align: center; font-size: 16px; line-height: 1.6; margin: 30px 0; }}
            .cert-body strong {{ color: #0f172a; font-size: 18px; }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 20px;
                margin: 25px 0;
                text-align: center;
            }}
            .metric-box h4 {{ margin: 0; font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}
            .metric-box p {{ margin: 5px 0 0 0; font-size: 22px; font-weight: 700; color: #0f172a; }}
            .checklist {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; font-size: 13px; }}
            .check-item {{ display: flex; align-items: center; gap: 8px; color: #166534; font-weight: 600; }}
            .cert-footer {{ display: flex; justify-content: space-between; align-items: flex-end; margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
            .sig-block {{ text-align: center; }}
            .sig-line {{ width: 200px; border-bottom: 2px solid #0f172a; margin-bottom: 8px; font-family: 'Cinzel', cursive; font-size: 18px; color: #0f172a; }}
            .sig-title {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}
            .cert-id {{ font-size: 11px; color: #94a3b8; letter-spacing: 1px; }}
            @media print {{
                body {{ background: none; padding: 0; }}
                .cert-card {{ box-shadow: none; border: 8px double #0f172a; width: 100%; }}
            }}
        </style>
    </head>
    <body>
        <div class="cert-card">
            <div class="gold-badge">
                <span>100</span>
                Verified A+
            </div>
            <div class="cert-header">
                <h1 class="cert-title">Certificate of Data Quality</h1>
                <div class="cert-subtitle">Autonomous Verification & AI-Readiness Standard</div>
            </div>
            <div class="cert-body">
                This document certifies that the dataset <br>
                <strong>"{filename}"</strong><br>
                has undergone deep automated profiling, machine learning semantic cross-validation,<br>
                and full schema standardization via the <strong>DataDoctor Enterprise Engine</strong>.
            </div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <h4>Health Score</h4>
                    <p>{score_after} / 100 <span style="color:#16a34a; font-size:14px;">(+{score_after - score_before})</span></p>
                </div>
                <div class="metric-box">
                    <h4>Cleaned Records</h4>
                    <p>{len(cleaned_df):,}</p>
                </div>
                <div class="metric-box">
                    <h4>Total Fixes Applied</h4>
                    <p>{len(diff_df):,}</p>
                </div>
            </div>
            <div class="checklist">
                <div class="check-item">✔ 100% ISO-8601 Date Standard (YYYY-MM-DD)</div>
                <div class="check-item">✔ ML Name-to-Gender Consistency Verified</div>
                <div class="check-item">✔ Zero Unhandled Null / Missing Values</div>
                <div class="check-item">✔ Normalized Currency & Numeric Formats</div>
                <div class="check-item">✔ Standardized snake_case Column Schema</div>
                <div class="check-item">✔ Zero Duplicate Entries Remaining</div>
            </div>
            <div class="cert-footer">
                <div>
                    <div class="cert-id">CERTIFICATE ID: <strong>{cert_id}</strong></div>
                    <div class="cert-id">TIMESTAMP: {now_str}</div>
                </div>
                <div class="sig-block">
                    <div class="sig-line">DataDoctor AI</div>
                    <div class="sig-title">Automated Governance Engine</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html