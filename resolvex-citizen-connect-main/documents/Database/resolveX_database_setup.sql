-- ========================================
-- ResolveX Grievance Management System
-- MySQL Database Schema
-- 21 Tables across 6 categories
-- ========================================

-- Create Database
CREATE DATABASE IF NOT EXISTS resolveX_grievance_system;
USE resolveX_grievance_system;

-- ========================================
-- CATEGORY 1: CORE ENTITIES (5 tables)
-- ========================================

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    phone VARCHAR(15),
    role ENUM('citizen','department','admin') NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    profile_picture VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_role (role)
);

-- 2. Departments Table
CREATE TABLE IF NOT EXISTS departments (
    department_id INT AUTO_INCREMENT PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    contact_email VARCHAR(100),
    contact_phone VARCHAR(15),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_department_name (department_name)
);

-- 3. Locations Table (States and Districts)
CREATE TABLE IF NOT EXISTS locations (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    state VARCHAR(50) NOT NULL,
    district VARCHAR(50) NOT NULL,
    latitude DOUBLE,
    longitude DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_state_district (state, district),
    INDEX idx_state (state),
    INDEX idx_district (district)
);

-- 4. Complaint Types Table
CREATE TABLE IF NOT EXISTS complaint_types (
    type_id INT AUTO_INCREMENT PRIMARY KEY,
    type_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    department_id INT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    INDEX idx_department_id (department_id)
);

-- 5. User Locations Junction Table
CREATE TABLE IF NOT EXISTS user_locations (
    user_location_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    location_id INT NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (location_id) REFERENCES locations(location_id),
    UNIQUE KEY uk_user_location (user_id, location_id)
);

-- ========================================
-- CATEGORY 2: COMPLAINT PROCESSING (5 tables)
-- ========================================

-- 6. Complaints Table (Main Table)
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_number VARCHAR(50) UNIQUE NOT NULL,
    user_id INT NOT NULL,
    department_id INT NOT NULL,
    complaint_type_id INT,
    location_id INT,
    language VARCHAR(20) DEFAULT 'English',
    complaint_text LONGTEXT NOT NULL,
    status ENUM('Registered','In Progress','Escalated','Resolved','Closed','Rejected') DEFAULT 'Registered',
    priority ENUM('Normal','Urgent') DEFAULT 'Normal',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    resolved_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (complaint_type_id) REFERENCES complaint_types(type_id),
    FOREIGN KEY (location_id) REFERENCES locations(location_id),
    INDEX idx_complaint_number (complaint_number),
    INDEX idx_user_id (user_id),
    INDEX idx_department_id (department_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_priority (priority)
);

-- 7. Complaint Status History Table
CREATE TABLE IF NOT EXISTS complaint_status_history (
    status_history_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    previous_status VARCHAR(50),
    new_status VARCHAR(50) NOT NULL,
    changed_by INT,
    comments TEXT,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    FOREIGN KEY (changed_by) REFERENCES users(user_id),
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_changed_at (changed_at)
);

-- 8. Complaint Assignments Table
CREATE TABLE IF NOT EXISTS complaint_assignments (
    assignment_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    assigned_to INT,
    assigned_by INT,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    FOREIGN KEY (assigned_to) REFERENCES users(user_id),
    FOREIGN KEY (assigned_by) REFERENCES users(user_id),
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_assigned_to (assigned_to)
);

-- 9. Complaint Comments Table
CREATE TABLE IF NOT EXISTS complaint_comments (
    comment_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    user_id INT NOT NULL,
    comment_text TEXT NOT NULL,
    is_public BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);

-- 10. Complaint Attachments Table
CREATE TABLE IF NOT EXISTS complaint_attachments (
    attachment_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    file_size INT,
    file_type VARCHAR(50),
    uploaded_by INT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_by) REFERENCES users(user_id),
    INDEX idx_complaint_id (complaint_id)
);

-- ========================================
-- CATEGORY 3: AI & NLP LOGS (3 tables)
-- ========================================

-- 11. Sentiment Analysis Table
CREATE TABLE IF NOT EXISTS sentiment_analysis (
    sentiment_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    sentiment_label ENUM('Positive','Negative','Neutral') NOT NULL,
    sentiment_score FLOAT CHECK (sentiment_score BETWEEN 0.0 AND 1.0),
    model_version VARCHAR(50),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_sentiment_label (sentiment_label)
);

-- 12. Priority Analysis Table
CREATE TABLE IF NOT EXISTS priority_analysis (
    priority_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    urgency_score FLOAT CHECK (urgency_score BETWEEN 0.0 AND 1.0),
    priority_level ENUM('Normal','Urgent') NOT NULL,
    priority_reasons TEXT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_priority_level (priority_level)
);

-- 13. NLP Processing Logs Table
CREATE TABLE IF NOT EXISTS nlp_processing_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    processing_type VARCHAR(100),
    keywords_extracted TEXT,
    entity_locations TEXT,
    processing_time FLOAT,
    model_used VARCHAR(100),
    status ENUM('Success','Failed','Pending') DEFAULT 'Pending',
    error_message TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_status (status)
);

-- ========================================
-- CATEGORY 4: TRACKING & NOTIFICATIONS (3 tables)
-- ========================================

-- 14. Notifications Table
CREATE TABLE IF NOT EXISTS notifications (
    notification_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    complaint_id INT,
    notification_type VARCHAR(100),
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    read_at DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_is_read (is_read),
    INDEX idx_created_at (created_at)
);

-- 15. Complaint Tracking Table
CREATE TABLE IF NOT EXISTS complaint_tracking (
    tracking_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    tracking_stage VARCHAR(100),
    stage_description TEXT,
    started_at TIMESTAMP,
    completed_at DATETIME,
    estimated_completion DATE,
    progress_percentage INT,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    INDEX idx_complaint_id (complaint_id)
);

-- 16. Status Updates Table
CREATE TABLE IF NOT EXISTS status_updates (
    update_id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    update_title VARCHAR(255),
    update_description TEXT,
    updated_by INT,
    visibility ENUM('Public','Private','Department') DEFAULT 'Public',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE,
    FOREIGN KEY (updated_by) REFERENCES users(user_id),
    INDEX idx_complaint_id (complaint_id),
    INDEX idx_created_at (created_at)
);

-- ========================================
-- CATEGORY 5: ANALYTICS & HEATMAP (3 tables)
-- ========================================

-- 17. Heatmap Cache Table (for Google Maps integration)
CREATE TABLE IF NOT EXISTS heatmap_cache (
    heatmap_id INT AUTO_INCREMENT PRIMARY KEY,
    location_id INT,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    complaint_count INT DEFAULT 0,
    complaint_type_distribution JSON,
    average_sentiment_score FLOAT,
    average_urgency_score FLOAT,
    severity_level ENUM('Low','Medium','High','Critical') DEFAULT 'Low',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (location_id) REFERENCES locations(location_id),
    INDEX idx_latitude_longitude (latitude, longitude),
    INDEX idx_last_updated (last_updated)
);

-- 18. Analytics Dashboard Table
CREATE TABLE IF NOT EXISTS analytics_dashboard (
    analytics_id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value INT,
    metric_type ENUM('Total','By_Department','By_Status','By_Priority','By_Language','By_Sentiment'),
    time_period VARCHAR(50),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_type (metric_type),
    INDEX idx_time_period (time_period)
);

-- 19. Complaint Statistics Table
CREATE TABLE IF NOT EXISTS complaint_statistics (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    stat_date DATE NOT NULL,
    department_id INT,
    total_complaints INT DEFAULT 0,
    resolved_complaints INT DEFAULT 0,
    pending_complaints INT DEFAULT 0,
    urgent_complaints INT DEFAULT 0,
    average_resolution_time INT,
    satisfaction_rating FLOAT,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    UNIQUE KEY uk_date_dept (stat_date, department_id),
    INDEX idx_stat_date (stat_date),
    INDEX idx_department_id (department_id)
);

-- ========================================
-- CATEGORY 6: SECURITY & AUDIT (2 tables)
-- ========================================

-- 20. Audit Logs Table
CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action_type VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id INT,
    old_value TEXT,
    new_value TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    action_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_user_id (user_id),
    INDEX idx_action_timestamp (action_timestamp),
    INDEX idx_resource_type (resource_type)
);

-- 21. User Sessions Table
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_token VARCHAR(500),
    ip_address VARCHAR(45),
    user_agent TEXT,
    login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    logout_at DATETIME,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_token (session_token),
    INDEX idx_login_at (login_at)
);

-- ========================================
-- CREATE INDEXES FOR PERFORMANCE
-- ========================================

-- Full-text search index for complaints
ALTER TABLE complaints ADD FULLTEXT INDEX ft_complaint_text (complaint_text);

-- Composite indexes for common queries
CREATE INDEX idx_complaints_dept_date ON complaints(department_id, created_at);
CREATE INDEX idx_complaints_status_priority ON complaints(status, priority);
CREATE INDEX idx_complaints_user_status ON complaints(user_id, status);

-- ========================================
-- SAMPLE DATA (Optional - for testing)
-- ========================================

-- Insert sample departments
INSERT INTO departments (department_name, description, contact_email) VALUES
('Education Services', 'Education Department - Schools, Colleges, Training', 'edu@resolve.gov'),
('Health Services', 'Health Department - Hospitals, Clinics, Medical Services', 'health@resolve.gov'),
('Municipal Administration', 'Municipal Administration - Birth/Death Certificates, Permits', 'municipal@resolve.gov'),
('Public Works', 'Public Works Department - Roads, Infrastructure, Maintenance', 'publicworks@resolve.gov'),
('Transport Services', 'Transport Services - Bus, Auto, Traffic Issues', 'transport@resolve.gov'),
('Water Supply', 'Water Supply Department - Water Connections, Leaks', 'water@resolve.gov'),
('Electricity Department', 'Electricity - Power Connections, Outages', 'electricity@resolve.gov'),
('Sanitation & Waste Management', 'Sanitation - Waste Collection, Cleanliness', 'sanitation@resolve.gov');

-- Insert sample locations (States and Districts)
INSERT INTO locations (state, district, latitude, longitude) VALUES
('Tamil Nadu', 'Trichy', 10.7905, 78.7047),
('Tamil Nadu', 'Coimbatore', 11.0081, 76.9025),
('Tamil Nadu', 'Salem', 11.6643, 78.1460),
('Tamil Nadu', 'Karur', 10.9299, 78.1724),
('Tamil Nadu', 'Erode', 11.3410, 77.7172),
('Tamil Nadu', 'Chennai', 13.0827, 80.2707),
('Tamil Nadu', 'Madurai', 9.9252, 78.1198),
('Tamil Nadu', 'Tiruppur', 11.1085, 77.3411);

-- ========================================
-- END OF DATABASE SCHEMA
-- ========================================