def get_db_connection():
    import mysql.connector
    from config import Config

    connection = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB
    )
    return connection

def register_user(name, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
    conn.commit()
    cursor.close()
    conn.close()

def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def login_user(email, password):
    user = get_user_by_email(email)
    if user and user['password'] == password:
        return user
    return None

def log_emotion(user_id, emotion):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotion_logs (user_id, emotion) VALUES (%s, %s)", (user_id, emotion))
    conn.commit()
    cursor.close()
    conn.close()

def get_emotion_trends(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT emotion, COUNT(*) as count FROM emotion_logs WHERE user_id = %s GROUP BY emotion", (user_id,))
    trends = cursor.fetchall()
    cursor.close()
    conn.close()
    return trends

def get_recent_emotion_logs(user_id, limit=25):
    """Get the most recent emotion logs for a user, ordered by timestamp"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT emotion, timestamp 
        FROM emotion_logs 
        WHERE user_id = %s 
        ORDER BY timestamp DESC 
        LIMIT %s
    """, (user_id, limit))
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    # Reverse to get chronological order (oldest to newest)
    return logs[::-1] if logs else []

def get_user_stats(user_id):
    """Get comprehensive user statistics for dashboard"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get total sessions count
    cursor.execute("SELECT COUNT(*) as total_sessions FROM emotion_logs WHERE user_id = %s", (user_id,))
    total_sessions = cursor.fetchone()['total_sessions']
    
    # Get days active (distinct dates)
    cursor.execute("""
        SELECT COUNT(DISTINCT DATE(timestamp)) as days_active 
        FROM emotion_logs 
        WHERE user_id = %s
    """, (user_id,))
    days_active = cursor.fetchone()['days_active']
    
    # Calculate happiness progress (percentage of positive emotions)
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN emotion IN ('happy', 'surprise') THEN 1 ELSE 0 END) as positive
        FROM emotion_logs 
        WHERE user_id = %s
    """, (user_id,))
    emotion_stats = cursor.fetchone()
    
    happiness_progress = 0
    if emotion_stats['total'] > 0:
        happiness_progress = round((emotion_stats['positive'] / emotion_stats['total']) * 100)
    
    cursor.close()
    conn.close()
    
    return {
        'total_sessions': total_sessions,
        'days_active': days_active,
        'happiness_progress': happiness_progress
    }

def get_recent_sessions_with_details(user_id, limit=10):
    """Get recent emotion logs with proper formatting for dashboard table"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT 
            emotion, 
            timestamp,
            DATE_FORMAT(timestamp, '%Y-%m-%d %H:%i') as formatted_datetime,
            DATE_FORMAT(timestamp, '%M/%d/%Y at %h:%i %p') as display_datetime
        FROM emotion_logs 
        WHERE user_id = %s 
        ORDER BY timestamp DESC 
        LIMIT %s
    """, (user_id, limit))
    sessions = cursor.fetchall()
    cursor.close()
    conn.close()
    return sessions

def delete_emotion_by_timestamp(user_id, timestamp):
    """Delete a specific emotion log by timestamp for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM emotion_logs 
        WHERE user_id = %s AND timestamp = %s
    """, (user_id, timestamp))
    deleted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    return deleted_count

def delete_all_emotions(user_id):
    """Delete all emotion logs for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM emotion_logs WHERE user_id = %s", (user_id,))
    deleted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    return deleted_count