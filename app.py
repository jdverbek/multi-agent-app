import asyncio
import os
from flask import Flask, request, jsonify, render_template
from main_controller import MainController
from tasks import Task

app = Flask(__name__)

# Global controller instance
controller = None
controller_task = None

def get_controller():
    """Get or create the global controller instance."""
    global controller, controller_task
    if controller is None:
        controller = MainController()
        # Start the controller loop in the background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        controller_task = loop.create_task(controller.run())
    return controller

@app.route('/')
def home():
    """Serve the web interface."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Get application status."""
    return jsonify({
        "status": "running",
        "controller_active": controller is not None,
        "available_agents": ["Manager", "CodeVerifier", "Developer"]
    })

@app.route('/submit', methods=['POST'])
def submit_task():
    """Submit a task to the multi-agent system."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        task_type = data.get('type', 'general')
        content = data.get('content', '')
        role = data.get('role', 'Developer')
        
        if not content:
            return jsonify({"error": "Content is required"}), 400
        
        # Create and submit task
        task = Task(type=task_type, content=content, role=role)
        controller = get_controller()
        
        # Submit task asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(controller.submit_task(task))
        
        # Wait a bit for processing (in a real app, you'd want proper async handling)
        loop.run_until_complete(asyncio.sleep(1))
        
        return jsonify({
            "status": "submitted",
            "task": {
                "type": task_type,
                "content": content,
                "role": role,
                "response": getattr(task, 'response', 'Processing...')
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

