from flask import Blueprint, render_template, session, redirect, url_for
from auth import login_required
from database import Database

history_bp = Blueprint('history', __name__)
db = Database()

@history_bp.route('/history')
@login_required
def view_history():
    # Get user's history
    history = db.get_user_history(session['user_id'])
    
    # Format history entries for display
    formatted_history = []
    for entry in history:
        formatted_entry = {
            'id': entry['id'],
            'timestamp': entry['created_at'],
            'action': entry['action_type'],
            'details': entry['action_details']
        }
        
        # Add human-readable description based on action type
        if entry['action_type'] == 'login':
            formatted_entry['description'] = f"Logged in with email {entry['action_details'].get('email')}"
        elif entry['action_type'] == 'logout':
            formatted_entry['description'] = f"Logged out"
        elif entry['action_type'] == 'registration':
            formatted_entry['description'] = f"Registered account with email {entry['action_details'].get('email')}"
        elif entry['action_type'] == 'password_change':
            formatted_entry['description'] = f"Changed password"
        elif entry['action_type'] == 'create_dashboard':
            formatted_entry['description'] = f"Created dashboard '{entry['action_details'].get('name')}'"
        elif entry['action_type'] == 'upload_dataset':
            formatted_entry['description'] = f"Uploaded dataset '{entry['action_details'].get('name')}'"
        elif entry['action_type'] == 'create_visualization':
            formatted_entry['description'] = f"Created {entry['action_details'].get('type')} visualization"
        elif entry['action_type'] == 'update_dashboard_layout':
            formatted_entry['description'] = f"Updated dashboard layout"
        elif entry['action_type'] == 'delete_visualization':
            formatted_entry['description'] = f"Deleted visualization"
        else:
            formatted_entry['description'] = f"{entry['action_type']}"
        
        formatted_history.append(formatted_entry)
    
    return render_template('history.html', history=formatted_history)
