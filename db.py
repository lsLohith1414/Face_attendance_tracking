import sqlite3
import openpyxl
from datetime import datetime
import os
import time

# Database file path
DATABASE = os.environ.get('DATABASE_PATH', "employees.db")

def init_db():
    """Initialize the database. Create the table only if it doesn't exist."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Create employees table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS employees (
                    name TEXT NOT NULL,
                    emp_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    department TEXT NOT NULL
                )''')
    
    # Create attendance table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emp_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    check_in_time DATETIME NOT NULL,
                    check_out_time DATETIME,
                    total_working_hours REAL,
                    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
                )''')
    
    # Create total_working_hours table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS total_working_hours (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    employee_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_hours_worked REAL NOT NULL,
                    FOREIGN KEY (employee_id) REFERENCES employees(emp_id),
                    UNIQUE(employee_id, date)
                )''')
    
    conn.commit()
    conn.close()

def add_employee(name, emp_id, image_path, username, password, age, department):
    """Add an employee to the database."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    try:
        c.execute("""INSERT INTO employees 
                    (name, emp_id, image_path, username, password, age, department) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                   (name, emp_id, image_path, username, password, age, department))
        conn.commit()
        print(f"Employee added: {name} (ID: {emp_id})")
    except sqlite3.IntegrityError as e:
        print(f"Error adding employee: {e}")
        conn.rollback()
        return False
    
    conn.close()
    return True

def get_employee(emp_id):
    """Check if an employee exists in the database."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    c.execute("SELECT * FROM employees WHERE emp_id=?", (emp_id,))
    employee = c.fetchone()
    
    conn.close()
    return employee  # Returns employee details if found, otherwise None

def calculate_working_hours(check_in_time, check_out_time):
    """Calculate working hours between check-in and check-out times and return in HH:MM:SS format."""
    try:
        # Convert string times to datetime objects
        check_in_dt = datetime.strptime(check_in_time, "%Y-%m-%d %H:%M:%S")
        check_out_dt = datetime.strptime(check_out_time, "%Y-%m-%d %H:%M:%S")
        
        # Calculate time difference
        time_diff = check_out_dt - check_in_dt
        
        # Convert to hours, minutes, seconds
        total_seconds = int(time_diff.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        # Format as HH:MM:SS
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception as e:
        print(f"Error calculating working hours: {e}")
        return "00:00:00"

def update_total_working_hours(emp_id, name, date):
    """Update the total working hours for an employee on a specific date."""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Calculate total working hours
        total_hours = calculate_working_hours(emp_id, date)
        
        # Insert or update the total working hours
        c.execute("""
            INSERT INTO total_working_hours (employee_id, employee_name, date, total_hours_worked)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(employee_id, date) DO UPDATE SET
                total_hours_worked = excluded.total_hours_worked
        """, (emp_id, name, date, total_hours))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating total working hours: {e}")
        if conn:
            conn.close()
        return False

def sync_total_working_hours():
    """Synchronize total_working_hours table with aggregated data from attendance table."""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Get all attendance records grouped by employee and date
        c.execute("""
            SELECT 
                emp_id,
                name,
                date(check_in_time) as work_date,
                GROUP_CONCAT(total_working_hours) as hours_list
            FROM attendance
            WHERE check_out_time IS NOT NULL
            GROUP BY emp_id, date(check_in_time)
        """)
        
        records = c.fetchall()
        
        for record in records:
            emp_id, name, work_date, hours_list = record
            
            # Convert hours_list to list of time strings
            hours_list = hours_list.split(',')
            
            # Calculate total seconds from all working hours
            total_seconds = 0
            for time_str in hours_list:
                if time_str and time_str != "00:00:00":
                    try:
                        hours, minutes, seconds = map(int, time_str.split(':'))
                        total_seconds += (hours * 3600) + (minutes * 60) + seconds
                    except (ValueError, TypeError):
                        continue
            
            # Convert total seconds back to HH:MM:SS format
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            total_hours = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Update or insert into total_working_hours table
            c.execute("""
                INSERT INTO total_working_hours 
                    (employee_id, employee_name, date, total_hours_worked)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(employee_id, date) DO UPDATE SET
                    total_hours_worked = excluded.total_hours_worked
            """, (emp_id, name, work_date, total_hours))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error syncing total working hours: {e}")
        if conn:
            conn.close()
        return False

def update_attendance(emp_id, name, check_type):
    """Update the attendance database with check-in/check-out times."""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        # Get current date and time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        if check_type == "check_in":
            # Check if employee already checked in today
            c.execute("""
                SELECT id FROM attendance 
                WHERE emp_id = ? AND date(check_in_time) = date(?) AND check_out_time IS NULL
            """, (emp_id, current_time))
            
            if c.fetchone():
                conn.close()
                return False, "Already checked in. Please check out first."
            
            # Add new check-in entry
            c.execute("""
                INSERT INTO attendance (emp_id, name, check_in_time)
                VALUES (?, ?, ?)
            """, (emp_id, name, current_time))
            conn.commit()
            conn.close()
            return True, "Check-in successful"
            
        elif check_type == "check_out":
            # Check if employee has checked in today
            c.execute("""
                SELECT id, check_in_time FROM attendance 
                WHERE emp_id = ? AND date(check_in_time) = date(?) AND check_out_time IS NULL
            """, (emp_id, current_time))
            
            result = c.fetchone()
            if not result:
                conn.close()
                return False, "No active check-in found. Please check in first."
            
            # Calculate working hours in HH:MM:SS format
            working_hours = calculate_working_hours(result[1], current_time)
            
            # Update check-out time and total working hours
            c.execute("""
                UPDATE attendance 
                SET check_out_time = ?, total_working_hours = ?
                WHERE emp_id = ? AND date(check_in_time) = date(?) AND check_out_time IS NULL
            """, (current_time, working_hours, emp_id, current_time))
            
            conn.commit()
            conn.close()
            
            # Sync total working hours table
            sync_total_working_hours()
            
            return True, "Check-out successful"
            
    except Exception as e:
        print(f"Error updating attendance: {e}")
        if conn:
            conn.close()
        return False, f"Error updating attendance: {str(e)}"

def get_attendance_history(emp_id=None):
    """Get attendance history for an employee or all employees."""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        if emp_id:
            c.execute("""
                SELECT emp_id, name, check_in_time, check_out_time, total_working_hours
                FROM attendance
                WHERE emp_id = ?
                ORDER BY check_in_time DESC
            """, (emp_id,))
        else:
            c.execute("""
                SELECT emp_id, name, check_in_time, check_out_time, total_working_hours
                FROM attendance
                ORDER BY check_in_time DESC
            """)
            
        attendance_records = [{
            'emp_id': row[0],
            'name': row[1],
            'check_in_time': row[2],
            'check_out_time': row[3],
            'total_hours': row[4] if row[4] is not None else "00:00:00"
        } for row in c.fetchall()]
        
        conn.close()
        return attendance_records
    except Exception as e:
        print(f"Error getting attendance history: {e}")
        if conn:
            conn.close()
        return []

def get_daily_working_hours(emp_id=None, date=None):
    """Get daily working hours for an employee or all employees."""
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if emp_id:
            c.execute("""
                SELECT employee_id, employee_name, date, total_hours_worked
                FROM total_working_hours
                WHERE employee_id = ? AND date = ?
                ORDER BY date DESC
            """, (emp_id, date))
        else:
            c.execute("""
                SELECT employee_id, employee_name, date, total_hours_worked
                FROM total_working_hours
                WHERE date = ?
                ORDER BY employee_name
            """, (date,))
        
        records = [{
            'emp_id': row[0],
            'name': row[1],
            'date': row[2],
            'total_hours': row[3] if row[3] is not None else "00:00:00"
        } for row in c.fetchall()]
        
        conn.close()
        return records
    except Exception as e:
        print(f"Error getting daily working hours: {e}")
        if conn:
            conn.close()
        return []

# Initialize the database
init_db()