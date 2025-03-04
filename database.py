import sqlite3
import pandas as pd
import json
from datetime import datetime
import os
from config import DB_CONFIG

class Database:
    def __init__(self):
        # Create database file if it doesn't exist
        self.db_path = 'smart_dashboard.db'
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Datasets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Dashboards table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dashboards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            layout JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Visualizations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visualizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dashboard_id INTEGER NOT NULL,
            dataset_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            config JSON NOT NULL,
            position JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dashboard_id) REFERENCES dashboards (id),
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        ''')
        
        # History table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_details JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)
    
    def add_user(self, email, password_hash, name=None):
        """Add a new user to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (email, password_hash, name)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Email already exists
            return None
        finally:
            conn.close()
    
    def get_user_by_email(self, email):
        """Get user by email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'email': user[1],
                'password_hash': user[2],
                'name': user[3],
                'created_at': user[4]
            }
        return None
    
    def save_dataset(self, user_id, name, description, file_path):
        """Save dataset information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO datasets (user_id, name, description, file_path) VALUES (?, ?, ?, ?)",
            (user_id, name, description, file_path)
        )
        dataset_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return dataset_id
    
    def get_user_datasets(self, user_id):
        """Get all datasets for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        datasets = cursor.fetchall()
        conn.close()
        
        result = []
        for dataset in datasets:
            result.append({
                'id': dataset[0],
                'user_id': dataset[1],
                'name': dataset[2],
                'description': dataset[3],
                'file_path': dataset[4],
                'created_at': dataset[5]
            })
        return result
    
    def create_dashboard(self, user_id, name):
        """Create a new dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO dashboards (user_id, name, layout) VALUES (?, ?, ?)",
            (user_id, name, '{}')
        )
        dashboard_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log this action in history
        self.add_history(user_id, 'create_dashboard', {'dashboard_id': dashboard_id, 'name': name})
        
        return dashboard_id
    
    def get_user_dashboards(self, user_id):
        """Get all dashboards for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dashboards WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
        dashboards = cursor.fetchall()
        conn.close()
        
        result = []
        for dashboard in dashboards:
            result.append({
                'id': dashboard[0],
                'user_id': dashboard[1],
                'name': dashboard[2],
                'layout': json.loads(dashboard[3]) if dashboard[3] else {},
                'created_at': dashboard[4],
                'updated_at': dashboard[5]
            })
        return result
    
    def save_visualization(self, dashboard_id, dataset_id, viz_type, config, position):
        """Save a visualization to a dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO visualizations (dashboard_id, dataset_id, type, config, position) VALUES (?, ?, ?, ?, ?)",
            (dashboard_id, dataset_id, viz_type, json.dumps(config), json.dumps(position))
        )
        viz_id = cursor.lastrowid
        
        # Update dashboard's updated_at timestamp
        cursor.execute(
            "UPDATE dashboards SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), dashboard_id)
        )
        
        conn.commit()
        conn.close()
        return viz_id
    
    def get_dashboard_visualizations(self, dashboard_id):
        """Get all visualizations for a dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM visualizations WHERE dashboard_id = ?", (dashboard_id,))
        visualizations = cursor.fetchall()
        conn.close()
        
        result = []
        for viz in visualizations:
            result.append({
                'id': viz[0],
                'dashboard_id': viz[1],
                'dataset_id': viz[2],
                'type': viz[3],
                'config': json.loads(viz[4]),
                'position': json.loads(viz[5]) if viz[5] else {},
                'created_at': viz[6]
            })
        return result
    
    def add_history(self, user_id, action_type, action_details):
        """Add an entry to the history table"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (user_id, action_type, action_details) VALUES (?, ?, ?)",
            (user_id, action_type, json.dumps(action_details))
        )
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id, limit=50):
        """Get history for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", 
            (user_id, limit)
        )
        history = cursor.fetchall()
        conn.close()
        
        result = []
        for entry in history:
            result.append({
                'id': entry[0],
                'user_id': entry[1],
                'action_type': entry[2],
                'action_details': json.loads(entry[3]) if entry[3] else {},
                'created_at': entry[4]
            })
        return result
    
    def load_dataset(self, dataset_id):
        """Load a dataset as a pandas DataFrame"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM datasets WHERE id = ?", (dataset_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        file_path = result[0]
        if not os.path.exists(file_path):
            return None
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            return None
