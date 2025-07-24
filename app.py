from flask import Flask, render_template, request, jsonify, send_file, abort
import asyncio
import logging
import os
import time
from main_controller import MainController
from visual_flow_executor import VisualFlowExecutor, VisualBlock, VisualConnection

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
    """Execute a visual flow."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Parse visual flow data
        blocks_data = data.get('blocks', [])
        connections_data = data.get('connections', [])
        task_content = data.get('content', '')
        task_type = data.get('type', 'general')
        
        print(f"[DEBUG] Received blocks: {len(blocks_data)}, connections: {len(connections_data)}")
        print(f"[DEBUG] Task content: {task_content[:100]}...")
        
        if not blocks_data or not task_content:
            return jsonify({"error": "blocks and content are required"}), 400
        
        # Convert to visual flow objects with error handling
        visual_blocks = []
        for block in blocks_data:
            try:
                visual_block = VisualBlock(
                    id=block['id'],
                    type=block['type'],
                    x=block['x'],
                    y=block['y'],
                    config=block.get('config', {})
                )
                visual_blocks.append(visual_block)
                print(f"[DEBUG] Created block: {visual_block.id} ({visual_block.type})")
            except Exception as e:
                print(f"[ERROR] Failed to create block {block.get('id', 'unknown')}: {str(e)}")
                return jsonify({"error": f"Invalid block data: {str(e)}"}), 400
        
        visual_connections = []
        for conn in connections_data:
            try:
                visual_connection = VisualConnection(
                    id=conn['id'],
                    from_block=conn['from'],
                    to_block=conn['to'],
                    type=conn['type']
                )
                visual_connections.append(visual_connection)
                print(f"[DEBUG] Created connection: {visual_connection.from_block} -> {visual_connection.to_block} ({visual_connection.type})")
            except Exception as e:
                print(f"[ERROR] Failed to create connection {conn.get('id', 'unknown')}: {str(e)}")
                return jsonify({"error": f"Invalid connection data: {str(e)}"}), 400
        
        # Create task
        task = Task(type=task_type, content=task_content)
        print(f"[DEBUG] Created task: {task.type}")
        
        controller = get_controller()
        print(f"[DEBUG] Got controller: {type(controller)}")
        
        # Check if visual_executor exists
        if not hasattr(controller, 'visual_executor'):
            print("[ERROR] Controller missing visual_executor")
            return jsonify({"error": "Visual executor not initialized"}), 500
        
        print(f"[DEBUG] Visual executor: {type(controller.visual_executor)}")
        
        # Execute visual flow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("[DEBUG] Starting visual flow execution...")
        result = loop.run_until_complete(
            controller.visual_executor.execute_visual_flow(
                visual_blocks, visual_connections, task
            )
        )
        print(f"[DEBUG] Execution completed: {type(result)}")
        
        return jsonify({
            "status": "executed",
            "result": result
        })
        
    except Exception as e:
        print(f"[ERROR] Visual flow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
    """Download files created by the system."""
    try:
        # Security: Only allow downloading from /tmp directory with specific patterns
        if not filename.startswith('presentation_') or not (filename.endswith('.pptx') or filename.endswith('.txt')):
            abort(404)
        
        file_path = os.path.join('/tmp', filename)
        
        if not os.path.exists(file_path):
            abort(404)
        
        # Determine the correct mimetype
        if filename.endswith('.pptx'):
            mimetype = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        else:
            mimetype = 'text/plain'
        
        return send_file(file_path, as_attachment=True, download_name=filename, mimetype=mimetype)
    
    except Exception as e:
        print(f"[DOWNLOAD] Error downloading file {filename}: {e}")
        abort(500)

@app.route('/list_files')
def list_files():
    """List available files for download."""
    try:
        files = []
        tmp_dir = '/tmp'
        
        if os.path.exists(tmp_dir):
            for filename in os.listdir(tmp_dir):
                if filename.startswith('presentation_') and (filename.endswith('.pptx') or filename.endswith('.txt')):
                    file_path = os.path.join(tmp_dir, filename)
                    file_stats = os.stat(file_path)
                    files.append({
                        'filename': filename,
                        'size': file_stats.st_size,
                        'created': time.ctime(file_stats.st_ctime),
                        'download_url': f'/download/{filename}'
                    })
        
        return jsonify({'files': files})
    
    except Exception as e:
        print(f"[LIST_FILES] Error listing files: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

