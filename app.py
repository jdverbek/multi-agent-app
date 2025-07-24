import asyncio
import os
import json
import time
from flask import Flask, request, jsonify, render_template, send_file
from main_controller import MainController
from tasks import Task
from visual_flow_executor import VisualFlowExecutor, VisualBlock, VisualConnection
from background_tasks import task_manager

app = Flask(__name__)

# Configure Flask for longer timeouts
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

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
        
        # Initialize visual flow executor
        controller.visual_executor = VisualFlowExecutor(controller.agents)
    return controller

@app.route('/')
def home():
    """Serve the web interface."""
    return render_template('index.html')

@app.route('/flow_designer')
def flow_designer():
    """Serve the visual flow designer."""
    return render_template('flow_designer.html')

@app.route('/status')
def status():
    """Get application status."""
    controller = get_controller()
    return jsonify({
        "status": "running",
        "controller_active": controller is not None,
        "available_agents": controller.get_available_agents(),
        "available_chains": controller.get_available_chains()
    })

@app.route('/agents')
def get_agents():
    """Get list of available agents."""
    controller = get_controller()
    return jsonify({
        "agents": controller.get_available_agents()
    })

@app.route('/chains')
def get_chains():
    """Get list of available chains."""
    controller = get_controller()
    chains = controller.get_available_chains()
    chain_definitions = {}
    
    for chain_id in chains:
        chain_definitions[chain_id] = controller.get_chain_definition(chain_id)
    
    return jsonify({
        "chains": chains,
        "definitions": chain_definitions
    })

@app.route('/chains/<chain_id>')
def get_chain_definition(chain_id):
    """Get definition of a specific chain."""
    controller = get_controller()
    definition = controller.get_chain_definition(chain_id)
    
    if definition:
        return jsonify(definition)
    else:
        return jsonify({"error": "Chain not found"}), 404

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
        chain_id = data.get('chain_id')  # Optional chain execution
        
        if not content:
            return jsonify({"error": "Content is required"}), 400
        
        # Create task
        task = Task(
            type=task_type, 
            content=content, 
            role=role,
            chain_id=chain_id
        )
        
        controller = get_controller()
        
        # Submit task asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if chain_id:
            # Execute chain
            result = loop.run_until_complete(controller.execute_chain(chain_id, task))
            return jsonify({
                "status": "chain_executed",
                "chain_id": chain_id,
                "result": result
            })
        else:
            # Regular task submission
            loop.run_until_complete(controller.submit_task(task))
            
            # Wait a bit for processing
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

@app.route('/chains', methods=['POST'])
def create_chain():
    """Create a custom agent chain."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        chain_id = data.get('chain_id')
        blocks_config = data.get('blocks', [])
        feedback_config = data.get('feedback_loops', [])
        
        if not chain_id or not blocks_config:
            return jsonify({"error": "chain_id and blocks are required"}), 400
        
        controller = get_controller()
        success = controller.create_custom_chain(chain_id, blocks_config, feedback_config)
        
        if success:
            return jsonify({
                "status": "created",
                "chain_id": chain_id,
                "message": "Chain created successfully"
            })
        else:
            return jsonify({"error": "Failed to create chain"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/execute_chain', methods=['POST'])
def execute_chain():
    """Execute a specific agent chain."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        chain_id = data.get('chain_id')
        content = data.get('content', '')
        task_type = data.get('type', 'general')
        
        if not chain_id or not content:
            return jsonify({"error": "chain_id and content are required"}), 400
        
        # Create task for chain execution
        task = Task(type=task_type, content=content)
        
        controller = get_controller()
        
        # Execute chain
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(controller.execute_chain(chain_id, task))
        
        return jsonify({
            "status": "executed",
            "chain_id": chain_id,
            "result": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/execute_visual_flow', methods=['POST'])
def execute_visual_flow():
    """Start a visual flow execution in the background."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Parse visual flow data
        blocks_data = data.get('blocks', [])
        connections_data = data.get('connections', [])
        task_content = data.get('content', '')
        task_type = data.get('type', 'general')
        
        if not blocks_data or not task_content:
            return jsonify({"error": "blocks and content are required"}), 400
        
        # Convert to visual flow objects
        visual_blocks = [
            VisualBlock(
                id=block['id'],
                type=block['type'],
                x=block['x'],
                y=block['y'],
                config=block['config']
            )
            for block in blocks_data
        ]
        
        visual_connections = [
            VisualConnection(
                id=conn['id'],
                from_block=conn['from'],
                to_block=conn['to'],
                type=conn['type']
            )
            for conn in connections_data
        ]
        
        # Create task
        task = Task(type=task_type, content=task_content)
        
        controller = get_controller()
        
        # Check if this is a simple task (< 5 blocks) for immediate execution
        if len(visual_blocks) <= 4:
            # Execute immediately for simple tasks
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    controller.visual_executor.execute_visual_flow(
                        visual_blocks, visual_connections, task
                    )
                )
                loop.close()
                
                return jsonify({
                    "status": "executed",
                    "result": result
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            # Start background task for complex workflows
            task_id = task_manager.start_task(
                visual_blocks, visual_connections, task, controller
            )
            
            return jsonify({
                "status": "started",
                "task_id": task_id,
                "message": "Complex workflow started in background. Use /task_status/{task_id} to check progress."
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    """Get the status of a background task."""
    try:
        task_info = task_manager.get_task_status(task_id)
        
        if not task_info:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify(task_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tasks')
def list_tasks():
    """List all background tasks."""
    try:
        tasks = task_manager.list_tasks()
        return jsonify({"tasks": tasks})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running background task."""
    try:
        success = task_manager.cancel_task(task_id)
        
        if success:
            return jsonify({"status": "cancelled", "task_id": task_id})
        else:
            return jsonify({"error": "Task not found or not running"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/validate_visual_flow', methods=['POST'])
def validate_visual_flow():
    """Validate a visual flow."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Parse visual flow data
        blocks_data = data.get('blocks', [])
        connections_data = data.get('connections', [])
        
        # Convert to visual flow objects
        visual_blocks = [
            VisualBlock(
                id=block['id'],
                type=block['type'],
                x=block['x'],
                y=block['y'],
                config=block['config']
            )
            for block in blocks_data
        ]
        
        visual_connections = [
            VisualConnection(
                id=conn['id'],
                from_block=conn['from'],
                to_block=conn['to'],
                type=conn['type']
            )
            for conn in connections_data
        ]
        
        controller = get_controller()
        
        # Validate flow
        validation_result = controller.visual_executor.validate_visual_flow(
            visual_blocks, visual_connections
        )
        
        # Get statistics
        stats = controller.visual_executor.get_flow_statistics(
            visual_blocks, visual_connections
        )
        
        return jsonify({
            "validation": validation_result,
            "statistics": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from the tmp directory."""
    try:
        # Security check - only allow .pptx files
        if not filename.endswith('.pptx'):
            return jsonify({"error": "Only PowerPoint files are allowed"}), 403
        
        file_path = os.path.join('/tmp', filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_files')
def list_files():
    """List all available PowerPoint files."""
    try:
        files = []
        tmp_dir = '/tmp'
        
        for filename in os.listdir(tmp_dir):
            if filename.endswith('.pptx'):
                file_path = os.path.join(tmp_dir, filename)
                file_stat = os.stat(file_path)
                
                files.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'created': time.strftime('%a %b %d %H:%M:%S %Y', time.localtime(file_stat.st_ctime)),
                    'download_url': f'/download/{filename}'
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({"files": files})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

