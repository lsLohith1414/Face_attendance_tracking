from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
import os
from werkzeug.utils import secure_filename
import face_model
import db
import base64
import cv2
import numpy as np
from functools import wraps
import threading
import time
import logging
import traceback
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Initialize the face recognition system
face_system = face_model.FaceRecognitionSystem()

# Load employee data into face recognition system
def load_employee_data():
    try:
        conn = sqlite3.connect(db.DATABASE)
        c = conn.cursor()
        
        # Get all employees
        c.execute("SELECT name, emp_id, image_path FROM employees")
        employees = c.fetchall()
        
        for name, emp_id, image_path in employees:
            if os.path.exists(image_path):
                # Load and process the image
                image = cv2.imread(image_path)
                if image is not None:
                    # Get face embedding
                    faces = face_system._detect_faces_dnn(image) if face_system.detector_type == "dnn" else face_system._fallback_detect_faces(image)
                    if faces:
                        x, y, w, h = faces[0]
                        face_img = image[y:y+h, x:x+w]
                        embedding = face_system.get_face_embedding(face_img)
                        if embedding is not None:
                            # Store the embedding with employee name and ID
                            person_name = f"{name}_{emp_id}"
                            face_system.face_embeddings[person_name] = [embedding]
        
        conn.close()
        print("Employee data loaded into face recognition system")
    except Exception as e:
        print(f"Error loading employee data: {e}")

# Load employee data when application starts
load_employee_data()

# Admin credentials (in production, use a proper database)
ADMIN_CREDENTIALS = {
    'admin': '1234'  # username: password
}

# Global training status
training_status = {
    'in_progress': False,
    'progress': 0,
    'status': '',
    'details': '',
    'completed': False
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session.get('user_type') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def train_model_async(emp_name, emp_id, valid_files):
    """Train the model asynchronously and update status"""
    global training_status
    
    try:
        # Reset training status
        training_status.update({
            'in_progress': True,
            'progress': 0,
            'status': 'Processing employee data...',
            'details': 'Initializing training process...',
            'completed': False,
            'success': False,
            'error': None
        })
        
        # Process each image (20% progress)
        total_files = len(valid_files)
        processed_faces = 0
        failed_faces = 0
        
        # Create a copy of file contents to avoid closed file issues
        file_contents = []
        for file in valid_files:
            file.seek(0)  # Reset file pointer
            content = file.read()
            file_contents.append(content)
        
        for i, content in enumerate(file_contents):
            # Create a new BytesIO object for each file
            from io import BytesIO
            file_obj = BytesIO(content)
            result = face_system.process_uploaded_image(f"{emp_name}_{emp_id}", file_obj)
            
            if result["success"]:
                processed_faces += 1
            else:
                failed_faces += 1
                
            progress = int((i + 1) / total_files * 20)
            training_status['progress'] = progress
            training_status['details'] = f'Processing image {i + 1} of {total_files} (Success: {processed_faces}, Failed: {failed_faces})'
            time.sleep(0.1)  # Small delay to show progress
        
        # Check if we have enough successful face detections
        if processed_faces < 5:  # Minimum required faces
            training_status['status'] = 'Training failed!'
            training_status['details'] = f'Not enough valid faces detected. Only {processed_faces} faces were successfully processed.'
            training_status['error'] = 'Insufficient valid face images'
            training_status['completed'] = True
            return
        
        # Generate embeddings (40% progress)
        training_status['status'] = 'Generating face embeddings...'
        training_status['progress'] = 20
        training_status['details'] = 'Creating face embeddings for recognition...'
        
        if face_system.process_and_save_embeddings(f"{emp_name}_{emp_id}"):
            training_status['progress'] = 60
            training_status['status'] = 'Training completed successfully!'
            training_status['details'] = f'Successfully processed {processed_faces} faces and generated embeddings.'
            training_status['success'] = True
            training_status['completed'] = True
        else:
            training_status['status'] = 'Training failed!'
            training_status['details'] = 'Failed to generate face embeddings.'
            training_status['error'] = 'Failed to generate face embeddings'
            training_status['completed'] = True
            
    except Exception as e:
        training_status['status'] = 'Error during training!'
        training_status['details'] = str(e)
        training_status['error'] = str(e)
        training_status['completed'] = True
    finally:
        training_status['in_progress'] = False

@app.route('/')
def home():
    if session.get('user'):
        if session.get('user_type') == 'admin':
            return render_template("index.html")
        else:
            return redirect(url_for('employee_dashboard'))
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_type = request.form.get('user_type')
        username = request.form.get('username')
        password = request.form.get('password')
        
        if user_type == 'admin':
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                session['user'] = username
                session['user_type'] = 'admin'
                flash('Welcome back, Admin!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid admin credentials.', 'error')
        else:  # employee login
            # Check if employee exists in database and verify password
            conn = sqlite3.connect(db.DATABASE)
            c = conn.cursor()
            c.execute("SELECT name, emp_id, username, password FROM employees WHERE username=?", (username,))
            employee = c.fetchone()
            conn.close()
            
            if employee and employee[3] == password:  # In production, use proper password hashing
                session['user'] = username
                session['user_type'] = 'employee'
                session['emp_name'] = employee[0]
                session['emp_id'] = employee[1]
                flash(f'Welcome back, {employee[0]}!', 'success')
                return redirect(url_for('employee_dashboard'))
            else:
                flash('Invalid employee credentials.', 'error')
    
    return render_template("login.html")

@app.route('/employee_dashboard')
@login_required
def employee_dashboard():
    if session.get('user_type') != 'employee':
        flash('Access denied. Employee access required.', 'error')
        return redirect(url_for('home'))
        
    return render_template("employee_dashboard.html")

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/add_employee', methods=['GET', 'POST'])
@admin_required
def add_employee():
    if request.method == 'GET':
        return render_template("add_employee.html")
        
    try:
        name = request.form.get('name', '').strip()
        emp_id = request.form.get('emp_id', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        age = request.form.get('age', '').strip()
        department = request.form.get('department', '').strip()
        files = request.files.getlist("images")
        
        logger.info(f"Received employee registration request for: {name} (ID: {emp_id})")
        
        # Validate inputs
        if not all([name, emp_id, username, password, age, department]):
            missing_fields = [field for field, value in {
                'name': name, 'emp_id': emp_id, 'username': username,
                'password': password, 'age': age, 'department': department
            }.items() if not value]
            logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return jsonify({
                "success": False,
                "message": f"Please provide all required fields. Missing: {', '.join(missing_fields)}"
            })
        
        if not files:
            logger.warning("No files uploaded")
            return jsonify({
                "success": False,
                "message": "Please upload at least one image."
            })
            
        if len(files) < 10 or len(files) > 50:
            logger.warning(f"Invalid number of files: {len(files)}")
            return jsonify({
                "success": False,
                "message": "Please upload between 10 to 50 images."
            })
        
        # Validate file types
        valid_files = []
        for file in files:
            if file and allowed_file(file.filename):
                valid_files.append(file)
            else:
                logger.warning(f"Invalid file type: {file.filename}")
                return jsonify({
                    "success": False,
                    "message": f"Invalid file type: {file.filename}. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
                })
        
        if not valid_files:
            logger.warning("No valid files after validation")
            return jsonify({
                "success": False,
                "message": "No valid image files were uploaded."
            })
        
        # Create employee directory
        emp_folder = os.path.join(UPLOAD_FOLDER, f"{name}_{emp_id}")
        os.makedirs(emp_folder, exist_ok=True)
        logger.info(f"Created employee directory: {emp_folder}")
        
        # Check if employee already exists
        if db.get_employee(emp_id):
            logger.warning(f"Employee ID already exists: {emp_id}")
            return jsonify({
                "success": False,
                "message": "Employee ID already exists!"
            })
        
        # Save files
        for i, file in enumerate(valid_files):
            filename = secure_filename(file.filename)
            file_path = os.path.join(emp_folder, filename)
            file.save(file_path)
            logger.debug(f"Saved file {i+1}/{len(valid_files)}: {file_path}")
        
        # Add to database with additional fields
        try:
            success = db.add_employee(name, emp_id, emp_folder, username, password, age, department)
            if not success:
                logger.error("Failed to add employee to database")
                return jsonify({
                    "success": False,
                    "message": "Failed to add employee to database."
                })
            logger.info(f"Successfully added employee to database: {name} (ID: {emp_id})")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "success": False,
                "message": f"Database error: {str(db_error)}"
            })
        
        # Start training process in background
        thread = threading.Thread(
            target=train_model_async,
            args=(name, emp_id, valid_files)
        )
        thread.daemon = True
        thread.start()
        logger.info("Started face recognition model training in background")
        
        return jsonify({
            "success": True,
            "message": "Employee added successfully! Training process started."
        })

    except Exception as e:
        logger.error(f"Unexpected error in add_employee: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"An unexpected error occurred: {str(e)}"
        })

@app.route('/training_status')
@admin_required
def get_training_status():
    """Get the current status of the training process"""
    return jsonify(training_status)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        # Get the base64 image data from the form
        image_data = request.form['image']
        
        # Remove the header from the base64 string
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Convert the bytes to a numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"face_detected": False, "error": "Failed to decode image"})
        
        # Process the frame using the face recognition system
        processed_frame = face_system.process_frame(image)
        
        # Check if any faces were detected (by comparing with original frame)
        faces_detected = not np.array_equal(image, processed_frame)
        
        return jsonify({"face_detected": faces_detected})
    except Exception as e:
        print(f"Error in detect_face: {str(e)}")
        return jsonify({"face_detected": False, "error": str(e)})

@app.route('/recognize_webcam', methods=['POST'])
def recognize_webcam():
    try:
        # Get the base64 image data from the form
        image_data = request.form['image']
        
        # Remove the header from the base64 string
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Convert the bytes to a numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to decode image'
            })
        
        # Process the frame using the face recognition system
        processed_frame = face_system.process_frame(image)
        
        # Get the recognized person and confidence
        recognized_person = None
        max_confidence = 0
        
        # Extract face and get embedding
        faces = face_system._detect_faces_dnn(image) if face_system.detector_type == "dnn" else face_system._fallback_detect_faces(image)
        
        if faces:
            x, y, w, h = faces[0]
            face_img = image[y:y+h, x:x+w]
            embedding = face_system.get_face_embedding(face_img)
            
            if embedding is not None:
                recognized_person, max_confidence = face_system.match_face(embedding)
        
        if recognized_person:
            # Split the person name into name and emp_id
            name, emp_id = recognized_person.split("_")
            
            # Update attendance in database with check-in type
            success, message = db.update_attendance(emp_id, name, "check_in")
            
            return jsonify({
                'success': success,
                'message': message,
                'name': name,
                'emp_id': emp_id,
                'probability': max_confidence,
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face not recognized.',
            })
    except Exception as e:
        print(f"Error in recognize_webcam: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/manage_employees')
@admin_required
def manage_employees():
    try:
        # Get all employees from database
        conn = sqlite3.connect(db.DATABASE)
        c = conn.cursor()
        c.execute("SELECT name, emp_id, username, age, department, password FROM employees")
        employees = [{
            'name': row[0],
            'emp_id': row[1],
            'username': row[2],
            'age': row[3],
            'department': row[4],
            'password': row[5]
        } for row in c.fetchall()]
        conn.close()
        
        return render_template("manage_employees.html", employees=employees)
    except Exception as e:
        logger.error(f"Error fetching employees: {str(e)}")
        flash('Error fetching employee data.', 'error')
        return redirect(url_for('home'))

@app.route('/edit_employee', methods=['POST'])
@admin_required
def edit_employee():
    try:
        emp_id = request.form.get('emp_id')
        name = request.form.get('name')
        username = request.form.get('username')
        age = request.form.get('age')
        department = request.form.get('department')
        password = request.form.get('password')
        
        conn = sqlite3.connect(db.DATABASE)
        c = conn.cursor()
        
        # Update employee details
        c.execute("""
            UPDATE employees 
            SET name=?, username=?, age=?, department=?, password=?
            WHERE emp_id=?
        """, (name, username, age, department, password, emp_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Employee details updated successfully!'
        })
    except Exception as e:
        logger.error(f"Error updating employee: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error updating employee: {str(e)}'
        })

@app.route('/delete_employee', methods=['POST'])
@admin_required
def delete_employee():
    try:
        emp_id = request.form.get('emp_id')
        
        # Delete employee from database
        conn = sqlite3.connect(db.DATABASE)
        c = conn.cursor()
        c.execute("DELETE FROM employees WHERE emp_id=?", (emp_id,))
        conn.commit()
        conn.close()
        
        # Delete employee's face data
        emp_name = emp_id.split('_')[0] if '_' in emp_id else emp_id
        face_system.delete_person(emp_name)
        
        return jsonify({
            'success': True,
            'message': 'Employee deleted successfully!'
        })
    except Exception as e:
        logger.error(f"Error deleting employee: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error deleting employee: {str(e)}'
        })

@app.route('/face_recognition')
@login_required
def face_recognition():
    return render_template("face_recognition.html")

@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    if request.method == 'GET':
        return render_template('mark_attendance.html')
    
    try:
        # Get the check type from the request
        data = request.get_json()
        check_type = data.get('check_type')
        
        if not check_type or check_type not in ['check_in', 'check_out']:
            return jsonify({
                'success': False,
                'message': 'Invalid check type'
            }), 400
        
        # Get the base64 image data from the form
        image_data = data.get('image')
        if not image_data:
            return jsonify({
                'success': False,
                'message': 'No image data provided'
            }), 400
        
        # Remove the header from the base64 string
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Convert the bytes to a numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to decode image'
            })
        
        # Process the frame using the face recognition system
        processed_frame = face_system.process_frame(image)
        
        # Get the recognized person and confidence
        recognized_person = None
        max_confidence = 0
        
        # Extract face and get embedding
        faces = face_system._detect_faces_dnn(image) if face_system.detector_type == "dnn" else face_system._fallback_detect_faces(image)
        
        if faces:
            x, y, w, h = faces[0]
            face_img = image[y:y+h, x:x+w]
            embedding = face_system.get_face_embedding(face_img)
            
            if embedding is not None:
                recognized_person, max_confidence = face_system.match_face(embedding)
        
        if recognized_person and max_confidence >= 0.75:  # Confidence threshold
            # Split the person name into name and emp_id
            name, emp_id = recognized_person.split("_")
            
            # Update attendance in database
            success, message = db.update_attendance(emp_id, name, check_type)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': message,
                    'name': name,
                    'emp_id': emp_id,
                    'confidence': max_confidence
                })
            else:
                return jsonify({
                    'success': False,
                    'message': message
                })
        else:
            return jsonify({
                'success': False,
                'message': 'Face not recognized or confidence too low'
            })
            
    except Exception as e:
        print(f"Error in mark_attendance: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        # Get the base64 image data from the form
        image_data = request.form['image']
        
        # Remove the header from the base64 string
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Convert the bytes to a numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to decode image'
            })
        
        # Process the frame using the face recognition system
        processed_frame = face_system.process_frame(image)
        
        # Get the recognized person and confidence
        recognized_person = None
        max_confidence = 0
        
        # Extract face and get embedding
        faces = face_system._detect_faces_dnn(image) if face_system.detector_type == "dnn" else face_system._fallback_detect_faces(image)
        
        if faces:
            x, y, w, h = faces[0]
            face_img = image[y:y+h, x:x+w]
            embedding = face_system.get_face_embedding(face_img)
            
            if embedding is not None:
                recognized_person, max_confidence = face_system.match_face(embedding)
        
        if recognized_person and max_confidence >= 0.75:  # Confidence threshold
            return jsonify({
                'success': True,
                'name': recognized_person.split('_')[0],
                'emp_id': recognized_person.split('_')[1],
                'confidence': max_confidence
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face not recognized or confidence too low'
            })
            
    except Exception as e:
        print(f"Error in recognize_face: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/attendance_report')
@login_required
def attendance_report():
    if session.get('user_type') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('home'))
    return render_template('attendance_report.html')

@app.route('/get_attendance_data')
@login_required
def get_attendance_data():
    if session.get('user_type') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    # Get all attendance data and statistics
    report_data = get_attendance_report()
    
    return jsonify({
        'attendance_data': report_data['attendance_data'],
        'total_records': len(report_data['attendance_data']),
        'total_employees': report_data['total_employees'],
        'present_employees': report_data['present_employees'],
        'absent_employees': report_data['absent_employees'],
        'today': report_data['today']
    })

def get_attendance_report():
    conn = sqlite3.connect(db.DATABASE)
    cursor = conn.cursor()
    
    # Get total employee count
    cursor.execute("SELECT COUNT(*) FROM employees")
    total_employees = cursor.fetchone()[0]
    
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get present employees for today
    cursor.execute("""
        SELECT COUNT(DISTINCT employee_id) 
        FROM total_working_hours 
        WHERE date = ?
    """, (today,))
    present_employees = cursor.fetchone()[0]
    
    # Calculate absent employees
    absent_employees = total_employees - present_employees
    
    # Get attendance data
    query = """
    SELECT 
        employee_id,
        employee_name,
        date,
        total_hours_worked
    FROM total_working_hours
    ORDER BY date DESC, employee_name
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    attendance_data = []
    for row in rows:
        attendance_data.append({
            'emp_id': row[0],
            'name': row[1],
            'date': row[2],
            'total_hours': row[3] if row[3] is not None else "00:00:00"
        })
    
    conn.close()
    return {
        'attendance_data': attendance_data,
        'total_employees': total_employees,
        'present_employees': present_employees,
        'absent_employees': absent_employees,
        'today': today
    }

@app.route('/get_employee_attendance/<emp_id>')
@login_required
def get_employee_attendance(emp_id):
    # Verify that the logged-in employee is requesting their own attendance
    if session.get('user_type') == 'employee' and session.get('emp_id') != emp_id:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = sqlite3.connect(db.DATABASE)
        cursor = conn.cursor()
        
        # Get employee details
        cursor.execute("""
            SELECT name, department 
            FROM employees 
            WHERE emp_id = ?
        """, (emp_id,))
        employee = cursor.fetchone()
        
        if not employee:
            return jsonify({'error': 'Employee not found'}), 404
        
        # Get total working hours data
        cursor.execute("""
            SELECT date, total_hours_worked
            FROM total_working_hours
            WHERE employee_id = ?
            ORDER BY date DESC
            LIMIT 5
        """, (emp_id,))
        total_hours_data = cursor.fetchall()
        
        # Get detailed attendance history
        cursor.execute("""
            SELECT 
                check_in_time,
                check_out_time,
                total_working_hours
            FROM attendance 
            WHERE emp_id = ?
            ORDER BY check_in_time DESC
        """, (emp_id,))
        attendance_history = cursor.fetchall()
        
        history_data = []
        for record in attendance_history:
            history_data.append({
                'check_in_time': record[0],
                'check_out_time': record[1],
                'total_hours': record[2] if record[2] is not None else "00:00:00"
            })
        
        # Format total hours data
        total_hours_formatted = []
        for record in total_hours_data:
            total_hours_formatted.append({
                'date': record[0],
                'total_hours': record[1] if record[1] is not None else "00:00:00"
            })
        
        conn.close()
        
        return jsonify({
            'employee': {
                'name': employee[0],
                'department': employee[1]
            },
            'attendance_history': history_data,
            'total_hours_data': total_hours_formatted
        })
    except Exception as e:
        print(f"Error getting employee attendance: {str(e)}")
        if conn:
            conn.close()
        return jsonify({'error': 'Failed to fetch attendance history'}), 500

@app.route('/employee_profile')
@login_required
def employee_profile():
    if session.get('user_type') != 'employee':
        flash('Access denied. Employee access required.', 'error')
        return redirect(url_for('home'))
    
    try:
        conn = sqlite3.connect(db.DATABASE)
        cursor = conn.cursor()
        
        # Get employee details
        cursor.execute("""
            SELECT name, emp_id, username, age, department, image_path
            FROM employees 
            WHERE emp_id = ?
        """, (session.get('emp_id'),))
        employee = cursor.fetchone()
        
        if not employee:
            flash('Employee not found.', 'error')
            return redirect(url_for('employee_dashboard'))
        
        employee_data = {
            'name': employee[0],
            'emp_id': employee[1],
            'username': employee[2],
            'age': employee[3],
            'department': employee[4],
            'image_path': employee[5]
        }
        
        conn.close()
        return render_template('employee_profile.html', employee=employee_data)
    except Exception as e:
        print(f"Error fetching employee profile: {str(e)}")
        flash('Error loading profile data.', 'error')
        return redirect(url_for('employee_dashboard'))

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if session.get('user_type') != 'employee':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        emp_id = session.get('emp_id')
        name = request.form.get('name')
        username = request.form.get('username')
        age = request.form.get('age')
        password = request.form.get('password')
        
        conn = sqlite3.connect(db.DATABASE)
        cursor = conn.cursor()
        
        # Check if username is already taken by another employee
        cursor.execute("""
            SELECT emp_id FROM employees 
            WHERE username = ? AND emp_id != ?
        """, (username, emp_id))
        
        if cursor.fetchone():
            conn.close()
            return jsonify({
                'success': False,
                'message': 'Username already taken. Please choose another.'
            })
        
        # Update employee details
        if password:
            cursor.execute("""
                UPDATE employees 
                SET name = ?, username = ?, age = ?, password = ?
                WHERE emp_id = ?
            """, (name, username, age, password, emp_id))
        else:
            cursor.execute("""
                UPDATE employees 
                SET name = ?, username = ?, age = ?
                WHERE emp_id = ?
            """, (name, username, age, emp_id))
        
        conn.commit()
        conn.close()
        
        # Update session data
        session['emp_name'] = name
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully!'
        })
    except Exception as e:
        print(f"Error updating profile: {str(e)}")
        if conn:
            conn.close()
        return jsonify({
            'success': False,
            'message': f'Error updating profile: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)