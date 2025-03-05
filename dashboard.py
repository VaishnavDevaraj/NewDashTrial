from turtle import position
from flask import Blueprint, request, render_template, redirect, url_for, flash, session, jsonify
from auth import login_required
from database import Database
from visualization import VisualizationGenerator
import pandas as pd
import json
import os
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

dashboard_bp = Blueprint('dashboard', __name__)
db = Database()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dashboard_bp.route('/')
@login_required
def index():
    # Get user's dashboards
    dashboards = db.get_user_dashboards(session['user_id'])
    
    # Get user's datasets
    datasets = db.get_user_datasets(session['user_id'])
    
    return render_template('dashboard.html', dashboards=dashboards, datasets=datasets)

@dashboard_bp.route('/create_dashboard', methods=['POST'])
@login_required
def create_dashboard():
    name = request.form.get('dashboard_name', 'New Dashboard')
    
    # Create a new dashboard
    dashboard_id = db.create_dashboard(session['user_id'], name)
    
    if dashboard_id:
        flash(f'Dashboard "{name}" created successfully', 'success')
        return redirect(url_for('dashboard.view_dashboard', dashboard_id=dashboard_id))
    else:
        flash('Failed to create dashboard', 'danger')
        return redirect(url_for('dashboard.index'))

@dashboard_bp.route('/dashboard/<int:dashboard_id>')
@login_required
def view_dashboard(dashboard_id):
    # Get dashboard details
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dashboards WHERE id = ? AND user_id = ?", 
                  (dashboard_id, session['user_id']))
    dashboard = cursor.fetchone()
    conn.close()
    
    if not dashboard:
        flash('Dashboard not found or access denied', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get visualizations for this dashboard
    visualizations = db.get_dashboard_visualizations(dashboard_id)
    
    # Get user's datasets
    datasets = db.get_user_datasets(session['user_id'])
    
    # Format dashboard data
    dashboard_data = {
        'id': dashboard[0],
        'user_id': dashboard[1],
        'name': dashboard[2],
        'layout': json.loads(dashboard[3]) if dashboard[3] else {},
        'created_at': dashboard[4],
        'updated_at': dashboard[5]
    }
    
    return render_template(
        'dashboard_view.html', 
        dashboard=dashboard_data, 
        visualizations=visualizations,
        datasets=datasets
    )

@dashboard_bp.route('/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard.index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard.index'))
    
    if file and allowed_file(file.filename):
        # Create upload folder if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{session['user_id']}_{filename}")
        file.save(file_path)
        
        # Get dataset name and description
        name = request.form.get('dataset_name', filename)
        description = request.form.get('dataset_description', '')
        
        # Save dataset information to database
        dataset_id = db.save_dataset(session['user_id'], name, description, file_path)
        
        # Log this action in history
        db.add_history(session['user_id'], 'upload_dataset', {
            'dataset_id': dataset_id,
            'name': name,
            'filename': filename
        })
        
        flash(f'Dataset "{name}" uploaded successfully', 'success')
    else:
        flash('Invalid file type. Allowed types: csv, xlsx, json, sqlite, db', 'danger')
    
    return redirect(url_for('dashboard.index'))

@dashboard_bp.route('/api/dataset/<int:dataset_id>/summary')
@login_required
def get_dataset_summary(dataset_id):
    # Check if user has access to this dataset
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", 
                  (dataset_id, session['user_id']))
    dataset = cursor.fetchone()
    conn.close()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found or access denied'}), 403
    
    # Load the dataset
    df = db.load_dataset(dataset_id)
    
    if df is None:
        return jsonify({'error': 'Failed to load dataset'}), 500
    
    # Create visualization generator and get summary
    viz_gen = VisualizationGenerator(df)
    summary = viz_gen.get_data_summary()
    
    return jsonify(summary)

@dashboard_bp.route('/api/dataset/<int:dataset_id>/suggest_visualizations')
@login_required
def suggest_visualizations(dataset_id):
    # Check if user has access to this dataset
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", 
                  (dataset_id, session['user_id']))
    dataset = cursor.fetchone()
    conn.close()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found or access denied'}), 403
    
    # Load the dataset
    df = db.load_dataset(dataset_id)
    
    if df is None:
        return jsonify({'error': 'Failed to load dataset'}), 500
    
    # Create visualization generator and get suggestions
    viz_gen = VisualizationGenerator(df)
    suggestions = viz_gen.suggest_visualizations()
    
    return jsonify(suggestions)

@dashboard_bp.route('/api/dataset/<int:dataset_id>/insights')
@login_required
def get_dataset_insights(dataset_id):
    import json

    def custom_serializer(obj):
        """Convert unsupported object types to JSON serializable format."""
        if hasattr(obj, "__dict__"):  # Convert objects to dictionaries
            return obj.__dict__
        return str(obj)  # Convert other types to strings

    
    # Check if user has access to this dataset
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", 
                  (dataset_id, session['user_id']))
    dataset = cursor.fetchone()
    conn.close()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found or access denied'}), 403
    
    # Load the dataset
    df = db.load_dataset(dataset_id)
    
    if df is None:
        return jsonify({'error': 'Failed to load dataset'}), 500
    
    # Create visualization generator and get insights
    viz_gen = VisualizationGenerator(df)
    insights = viz_gen.get_data_insights()
    
    # Convert non-serializable objects before returning JSON
    json_safe_insights = json.loads(json.dumps(insights, default=custom_serializer))
    
    return jsonify(json_safe_insights)

@dashboard_bp.route('/api/create_visualization', methods=['POST'])
@login_required
def create_visualization():
    data = request.get_json()

    # Debugging Log
    print("Received Create Visualization Data:", data)

    dashboard_id = data.get('dashboard_id')
    dataset_id = data.get('dataset_id')
    viz_type = data.get('type')
    config = data.get('config', {})

    # Validate inputs
    if not dashboard_id or not dataset_id or not viz_type:
        print(f"❌ Missing Parameters - dashboard_id: {dashboard_id}, dataset_id: {dataset_id}, type: {viz_type}")
        return jsonify({'error': 'Missing required parameters'}), 400

    
    # Check if user has access to this dashboard and dataset
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM dashboards WHERE id = ? AND user_id = ?", 
                  (dashboard_id, session['user_id']))
    dashboard = cursor.fetchone()
    
    cursor.execute("SELECT * FROM datasets WHERE id = ? AND user_id = ?", 
                  (dataset_id, session['user_id']))
    dataset = cursor.fetchone()
    
    conn.close()
    
    if not dashboard or not dataset:
        return jsonify({'error': 'Dashboard or dataset not found or access denied'}), 403
    
    # Load the dataset
    df = db.load_dataset(dataset_id)
    
    if df is None:
        return jsonify({'error': 'Failed to load dataset'}), 500
    
    # Create the visualization
    viz_gen = VisualizationGenerator(df)
    viz_data = viz_gen.create_visualization(viz_type, config)
    
    if viz_data is None:
        return jsonify({'error': 'Failed to create visualization'}), 500
    
    # Save the visualization to the database
    viz_id = db.save_visualization(dashboard_id, dataset_id, viz_type, config, position)
    
    # Log this action in history
    db.add_history(session['user_id'], 'create_visualization', {
        'dashboard_id': dashboard_id,
        'dataset_id': dataset_id,
        'visualization_id': viz_id,
        'type': viz_type
    })
    
    return jsonify({
        'id': viz_id,
        'dashboard_id': dashboard_id,
        'dataset_id': dataset_id,
        'type': viz_type,
        'config': config,
        'position': position,
        'data': json.loads(viz_data)
    })

@dashboard_bp.route('/api/update_dashboard_layout', methods=['POST'])
@login_required
def update_dashboard_layout():
    data = request.json
    
    dashboard_id = data.get('dashboard_id')
    layout = data.get('layout', {})
    
    # Validate inputs
    if not dashboard_id:
        return jsonify({'error': 'Missing dashboard_id parameter'}), 400
    
    # Check if user has access to this dashboard
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM dashboards WHERE id = ? AND user_id = ?", 
                  (dashboard_id, session['user_id']))
    dashboard = cursor.fetchone()
    
    if not dashboard:
        conn.close()
        return jsonify({'error': 'Dashboard not found or access denied'}), 403
    
    # Update the dashboard layout
    cursor.execute(
        "UPDATE dashboards SET layout = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(layout), dashboard_id)
    )
    conn.commit()
    conn.close()
    
    # Log this action in history
    db.add_history(session['user_id'], 'update_dashboard_layout', {
        'dashboard_id': dashboard_id
    })
    
    return jsonify({'success': True})

@dashboard_bp.route('/api/delete_visualization/<int:viz_id>', methods=['POST'])
@login_required
def delete_visualization(viz_id):
    # Check if user has access to this visualization
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT v.id, v.dashboard_id, d.user_id 
        FROM visualizations v
        JOIN dashboards d ON v.dashboard_id = d.id
        WHERE v.id = ? AND d.user_id = ?
    """, (viz_id, session['user_id']))
    
    viz = cursor.fetchone()
    
    if not viz:
        conn.close()
        return jsonify({'error': 'Visualization not found or access denied'}), 403
    
    # Delete the visualization
    cursor.execute("DELETE FROM visualizations WHERE id = ?", (viz_id,))
    
    # Update dashboard's updated_at timestamp
    cursor.execute(
        "UPDATE dashboards SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (viz[1],)  # dashboard_id
    )
    
    conn.commit()
    conn.close()
    
    # Log this action in history
    db.add_history(session['user_id'], 'delete_visualization', {
        'visualization_id': viz_id,
        'dashboard_id': viz[1]
    })
    
    return jsonify({'success': True})


