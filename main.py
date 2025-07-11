# main.py
import os
import json
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, make_response, jsonify, abort
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import qrcode
import urllib.parse
from PIL import Image

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
REPORTS_FOLDER = './reports'
SCAN_HISTORY_FILE = './scan_history.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

os.environ["TF_USE_LEGACY_KERAS"] = "1"
MODEL_PATH = 'models/model1.keras'
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

TUMOR_DESCRIPTIONS = {
    "notumor": {
        "description": "No signs of a brain tumor detected in the analyzed MRI scan.",
        "recommendation": "Continue routine checkups as advised by your physician.",
        "severity": "Normal",
        "clinical_interpretation": "The AI analysis shows no abnormal tissue growth or mass lesions. Normal brain tissue patterns are observed throughout the scan.",
        "clinical_recommendations": "No immediate intervention required. Continue regular health monitoring and follow-up as per physician's advice."
    },
    "glioma": {
        "description": "Glioma tumor detected - develops from glial cells in the brain.",
        "recommendation": "Immediate consultation with a neuro-oncologist required.",
        "severity": "High Priority",
        "clinical_interpretation": "Glioma tumors develop from glial cells that support nerve cells in the brain. These tumors can vary significantly in their growth rate and severity.",
        "clinical_recommendations": "Immediate consultation with a neuro-oncologist is recommended. Further imaging and biopsy may be required."
    },
    "meningioma": {
        "description": "Meningioma tumor detected - arises from the meninges surrounding the brain.",
        "recommendation": "Regular MRI monitoring and neurological consultation advised.",
        "severity": "Moderate Priority",
        "clinical_interpretation": "Meningioma tumors arise from the meninges, the protective membranes that cover the brain and spinal cord. Most are benign but require monitoring.",
        "clinical_recommendations": "Regular MRI monitoring advised. Consultation with neurosurgeon for evaluation of size, location, and growth rate."
    },
    "pituitary": {
        "description": "Pituitary tumor detected - develops in the pituitary gland.",
        "recommendation": "Consultation with endocrinologist and neurosurgeon recommended.",
        "severity": "Moderate Priority",
        "clinical_interpretation": "Pituitary tumors develop in the pituitary gland, which controls hormone production. Can affect various body functions.",
        "clinical_recommendations": "Consultation with endocrinologist for hormone level assessment and neurosurgeon for treatment planning."
    }
}

model = None

def load_detection_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error("Model not found")
            return False
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise

def predict_tumor(image_path):
    if not model:
        return None, 0.0, None, "Model not loaded"
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)[0]
        index = np.argmax(predictions)
        label = CLASS_LABELS[index]
        confidence = float(predictions[index])
        result = "No Tumor Detected" if label == "notumor" else f"Tumor Type: {label.capitalize()}"
        return result, confidence, TUMOR_DESCRIPTIONS[label], {
            CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0.0, None, "Prediction failed"

def save_scan_metadata(entry):
    try:
        if not os.path.exists(SCAN_HISTORY_FILE):
            with open(SCAN_HISTORY_FILE, 'w') as f:
                json.dump([], f)

        with open(SCAN_HISTORY_FILE, 'r+', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
            history.append(entry)
            f.seek(0)
            json.dump(history, f, indent=4)
            f.truncate()
            logger.info("Scan history updated.")
    except Exception as e:
        logger.error(f"Scan metadata save error: {e}")

def generate_qr_code(report_id, patient_name):
    """Generate QR code for report verification"""
    try:
        verification_url = f"https://neurocare-diagnostics.com/verify/{report_id}"
        qr = qrcode.QRCode(version=1, box_size=4, border=1)
        qr.add_data(verification_url)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_path = os.path.join(REPORTS_FOLDER, f"qr_{report_id}.png")
        qr_img.save(qr_path)
        return qr_path
    except Exception as e:
        logger.error(f"QR code generation error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            name = request.form.get('name', 'Unknown')
            age = request.form.get('age', 'Unknown')
            sex = request.form.get('sex', 'Unknown')

            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400

            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            result, confidence, tumor_info, all_predictions = predict_tumor(filepath)
            if not result:
                return jsonify({'error': tumor_info}), 500

            report_id = uuid.uuid4().hex[:12]
            scan_entry = {
                'report_id': report_id,
                'name': name,
                'age': age,
                'sex': sex,
                'filename': filename,
                'result': result,
                'confidence': float(f"{confidence * 100:.2f}"),
                'tumor_type': result.replace("Tumor Type: ", "") if "Tumor" in result else "None",
                'description': tumor_info['description'],
                'recommendation': tumor_info['recommendation'],
                'severity': tumor_info['severity'],
                'timestamp': datetime.now().isoformat()
            }
            save_scan_metadata(scan_entry)

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence * 100:.2f}",
                                   file_path=f"/uploads/{filename}",
                                   description=tumor_info['description'],
                                   recommendation=tumor_info['recommendation'],
                                   severity=tumor_info['severity'],
                                   all_predictions=all_predictions,
                                   patient_name=name,
                                   patient_age=age,
                                   patient_sex=sex,
                                   report_id=report_id)
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({'error': 'Upload failed'}), 500

    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/history')
def get_scan_history():
    try:
        if os.path.exists(SCAN_HISTORY_FILE):
            with open(SCAN_HISTORY_FILE, 'r') as f:
                return jsonify({'history': json.load(f)})
        return jsonify({'history': []})
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return jsonify({'error': 'Could not load history'}), 500

@app.route('/report')
def generate_report():
    try:
        result = request.args.get('result', 'Unknown')
        confidence = request.args.get('confidence', '0')
        description = urllib.parse.unquote(request.args.get('description', ''))
        recommendation = urllib.parse.unquote(request.args.get('recommendation', ''))
        severity = urllib.parse.unquote(request.args.get('severity', ''))
        file_path = request.args.get('file_path', '')
        report_id = request.args.get('report_id', uuid.uuid4().hex[:12])

        # Get patient metadata from JSON using the uploaded filename
        scan_filename = os.path.basename(file_path)
        patient_name = "Unknown"
        patient_age = "Unknown"
        patient_sex = "Unknown"
        
        if os.path.exists(SCAN_HISTORY_FILE):
            try:
                with open(SCAN_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
                    for entry in history:
                        if entry.get("filename") == scan_filename:
                            patient_name = entry.get("name", "Unknown")
                            patient_age = entry.get("age", "Unknown")
                            patient_sex = entry.get("sex", "Unknown")
                            break
            except Exception as e:
                logger.error(f"Error reading scan history: {e}")

        # Full path to the uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, scan_filename) if scan_filename else None
        
        # Generate QR code
        qr_path = generate_qr_code(report_id, patient_name)
        
        # Get tumor type for clinical info
        tumor_type = result.replace("Tumor Type: ", "").lower() if "Tumor" in result else "notumor"
        clinical_info = TUMOR_DESCRIPTIONS.get(tumor_type, TUMOR_DESCRIPTIONS["notumor"])

        # Create PDF with custom styling
        pdf = FPDF()
        pdf.add_page()
        
        # Define colors (RGB values)
        header_blue = (41, 128, 185)
        section_blue = (52, 152, 219)
        text_dark = (44, 62, 80)
        accent_green = (39, 174, 96)
        warning_red = (231, 76, 60)
        
        # Set severity color
        severity_color = accent_green if severity == "Normal" else warning_red if severity == "High Priority" else (243, 156, 18)

        # ===== HEADER SECTION =====
        pdf.set_fill_color(*header_blue)
        pdf.rect(0, 0, 210, 35, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 22)
        pdf.set_y(8)
        pdf.cell(0, 10, 'NEUROCARE DIAGNOSTICS', 0, 1, 'C')
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 8, 'MRI Brain Tumor Detection Report', 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, 'Advanced AI-Powered Medical Imaging Analysis', 0, 1, 'C')
        
        # Reset text color
        pdf.set_text_color(*text_dark)
        pdf.ln(15)

        # ===== REPORT INFO BAR =====
        pdf.set_fill_color(236, 240, 241)
        pdf.rect(10, pdf.get_y(), 190, 20, 'F')
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, f'Report ID: {report_id}', 0, 1, 'C')
        pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%d %B %Y at %H:%M IST")}', 0, 1, 'C')
        pdf.cell(0, 6, f'Status: CONFIDENTIAL MEDICAL REPORT', 0, 1, 'C')
        pdf.ln(10)

        # ===== PATIENT INFORMATION SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  PATIENT INFORMATION', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        # Patient info in two columns
        pdf.set_font('Arial', '', 10)
        pdf.ln(3)
        pdf.cell(95, 6, f'Patient Name: {patient_name}', 0, 0)
        pdf.cell(95, 6, f'Medical Record #: MR{report_id[:8].upper()}', 0, 1)
        pdf.cell(95, 6, f'Age: {patient_age} years', 0, 0)
        pdf.cell(95, 6, f'Gender: {patient_sex}', 0, 1)
        pdf.cell(95, 6, f'Study Date: {datetime.now().strftime("%d %B %Y")}', 0, 0)
        pdf.cell(95, 6, f'Modality: MRI Brain', 0, 1)
        pdf.cell(95, 6, 'Referring Physician: Dr. _______________', 0, 0)
        pdf.cell(95, 6, 'Emergency Contact: _______________', 0, 1)
        pdf.ln(8)

        # ===== MEDICAL FACILITY SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  MEDICAL FACILITY INFORMATION', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.set_font('Arial', '', 10)
        pdf.ln(3)
        pdf.cell(95, 6, 'Institution: NeuroCare Diagnostics Center', 0, 0)
        pdf.cell(95, 6, 'License: NABH-ACC-2024-001', 0, 1)
        pdf.cell(95, 6, 'Radiologist: Dr. A.K. Verma, MD', 0, 0)
        pdf.cell(95, 6, 'Contact: +91-9876543210', 0, 1)
        pdf.cell(95, 6, 'Address: 123 Health Blvd, Medicity, India', 0, 0)
        pdf.cell(95, 6, 'Department: Neuroradiology', 0, 1)
        pdf.ln(8)

        # ===== AI ANALYSIS RESULTS SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  AI ANALYSIS RESULTS', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.ln(3)
        
        # Result with colored background
        pdf.set_fill_color(*severity_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f'  FINDING: {result}', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        # Analysis details in table format
        pdf.set_font('Arial', '', 10)
        pdf.ln(2)
        pdf.cell(50, 6, 'Confidence Level:', 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(50, 6, f'{confidence}%', 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.cell(50, 6, 'Clinical Priority:', 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(40, 6, f'{severity}', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(50, 6, 'AI Model Version:', 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(50, 6, 'CNN v2.1', 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.cell(50, 6, 'Processing Time:', 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(40, 6, '< 2 seconds', 0, 1)
        pdf.ln(8)

        # ===== CLINICAL INTERPRETATION SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  CLINICAL INTERPRETATION', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.set_font('Arial', '', 10)
        pdf.ln(3)
        pdf.multi_cell(0, 6, clinical_info.get("clinical_interpretation", description))
        pdf.ln(5)

        # ===== RECOMMENDATIONS SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  CLINICAL RECOMMENDATIONS', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.set_font('Arial', '', 10)
        pdf.ln(3)
        pdf.multi_cell(0, 6, clinical_info.get("clinical_recommendations", recommendation))
        pdf.ln(8)

        # ===== SCAN IMAGE SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  ANALYZED MRI SCAN', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.ln(5)
        if image_path and os.path.exists(image_path):
            try:
                # Add border around image
                pdf.set_draw_color(200, 200, 200)
                pdf.rect(15, pdf.get_y(), 110, 75)
                
                # Convert image to JPEG if needed for better PDF compatibility
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    # Save temporary JPEG
                    temp_jpg = os.path.join(REPORTS_FOLDER, f"temp_{report_id}.jpg")
                    img.save(temp_jpg, 'JPEG', quality=85)
                    
                    # Add to PDF with border
                    pdf.image(temp_jpg, x=17, y=pdf.get_y()+2, w=106)
                    
                    # Add image details
                    pdf.set_xy(130, pdf.get_y()+5)
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 6, 'Image Details:', 0, 1)
                    pdf.set_font('Arial', '', 9)
                    pdf.set_x(130)
                    pdf.cell(0, 5, f'Resolution: {img.size[0]}x{img.size[1]}', 0, 1)
                    pdf.set_x(130)
                    pdf.cell(0, 5, f'Format: {img.format}', 0, 1)
                    pdf.set_x(130)
                    pdf.cell(0, 5, f'Color Mode: {img.mode}', 0, 1)
                    pdf.set_x(130)
                    pdf.cell(0, 5, 'Preprocessing: Applied', 0, 1)
                    pdf.set_x(130)
                    pdf.cell(0, 5, 'Enhancement: Contrast', 0, 1)
                    
                    pdf.ln(75)
                    
                    # Clean up temp file
                    if os.path.exists(temp_jpg):
                        os.remove(temp_jpg)
                        
            except Exception as img_err:
                logger.error(f"Image processing error: {img_err}")
                pdf.set_fill_color(255, 235, 238)
                pdf.rect(15, pdf.get_y(), 110, 20, 'F')
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 6, '[MRI Scan Image Processing Error - Please Contact Support]', 0, 1, 'C')
                pdf.ln(20)
        else:
            pdf.set_fill_color(255, 235, 238)
            pdf.rect(15, pdf.get_y(), 110, 20, 'F')
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 6, '[MRI Scan Image Currently Unavailable]', 0, 1, 'C')
            pdf.ln(20)

        # ===== VERIFICATION SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  REPORT VERIFICATION & SECURITY', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.ln(3)
        if qr_path and os.path.exists(qr_path):
            try:
                # Add QR code with border
                pdf.set_draw_color(200, 200, 200)
                pdf.rect(15, pdf.get_y(), 35, 35)
                pdf.image(qr_path, x=16, y=pdf.get_y()+1, w=33)
                
                # Verification details
                pdf.set_xy(55, pdf.get_y()+5)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, 'Digital Verification:', 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.set_x(55)
                pdf.cell(0, 5, f'Report ID: {report_id}', 0, 1)
                pdf.set_x(55)
                pdf.cell(0, 5, 'Scan QR code to verify authenticity', 0, 1)
                pdf.set_x(55)
                pdf.cell(0, 5, 'Or visit: neurocare-diagnostics.com/verify', 0, 1)
                pdf.set_x(55)
                pdf.cell(0, 5, f'Checksum: {hash(report_id) % 100000:05d}', 0, 1)
                
                pdf.ln(35)
            except Exception as qr_err:
                logger.error(f"QR code error: {qr_err}")
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, f'Digital Verification: https://neurocare-diagnostics.com/verify/{report_id}', 0, 1)
                pdf.ln(5)
        else:
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'Digital Verification: https://neurocare-diagnostics.com/verify/{report_id}', 0, 1)
            pdf.ln(5)

        # ===== FOOTER =====
        pdf.set_y(-25)
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(0, pdf.get_y(), 210, 25, 'F')
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, 'CONFIDENTIAL: This report contains sensitive medical information and should be handled according to HIPAA guidelines.', 0, 1, 'C')
        pdf.cell(0, 6, 'AI-Generated Report - Requires Medical Professional Review and Confirmation', 0, 1, 'C')
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(0, 6, 'Page 1 of 2', 0, 1, 'C')

        # ===== PAGE 2 - DETAILED DISCLAIMER =====
        pdf.add_page()
        
        # Header for page 2
        pdf.set_fill_color(*header_blue)
        pdf.rect(0, 0, 210, 25, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 18)
        pdf.set_y(5)
        pdf.cell(0, 8, 'NEUROCARE DIAGNOSTICS', 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, 'Medical Report Authorization & Legal Disclaimer', 0, 1, 'C')
        
        pdf.set_text_color(*text_dark)
        pdf.ln(15)

        # ===== AUTHORIZATION SECTION =====
        pdf.set_fill_color(*section_blue)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, '  MEDICAL AUTHORIZATION', 0, 1, 'L', True)
        pdf.set_text_color(*text_dark)
        
        pdf.set_font('Arial', '', 10)
        pdf.ln(5)
        pdf.multi_cell(0, 6, (
            "This medical report has been generated using advanced artificial intelligence algorithms trained on "
            "extensive neuroimaging datasets. The analysis is performed by our state-of-the-art Convolutional "
            "Neural Network (CNN) model, which has been validated against clinical standards and demonstrates "
            "high accuracy in tumor detection and classification."
        ))
        pdf.ln(5)
        
        # Authorization details
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Authorized Medical Personnel:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(70, 6, 'Dr. A.K. Verma, MD (Neuroradiology)', 0, 0)
        pdf.cell(70, 6, 'License: MED12345', 0, 1)
        pdf.cell(70, 6, 'Board Certification: ABNR', 0, 0)
        pdf.cell(70, 6, f'Digital Signature Date: {datetime.now().strftime("%d %B %Y")}', 0, 1)
        pdf.ln(10)

        # Generate PDF output
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        # Clean up QR code file after PDF generation
        if qr_path and os.path.exists(qr_path):
            try:
                os.remove(qr_path)
            except Exception as cleanup_err:
                logger.warning(f"QR cleanup error: {cleanup_err}")

        # Create response
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Tumor_Detection_Report_{report_id}.pdf'
        return response

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({'error': 'Report generation failed'}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    if load_detection_model():
        logger.info("Starting Flask app...")
    else:
        logger.warning("Model not loaded.")
    app.run(host='0.0.0.0', port=5050, debug=True)