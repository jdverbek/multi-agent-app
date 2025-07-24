#!/usr/bin/env python3
"""
Main entry point for the multi-agent application
"""

import os
import sys

def main():
    """Main entry point"""
    # Check if we're running in production (Render) or development
    if os.environ.get('RENDER'):
        # Production: Use Gunicorn with proper configuration
        import subprocess
        
        port = os.environ.get('PORT', '5000')
        
        # Start with Gunicorn using our configuration
        cmd = [
            'gunicorn',
            '--config', 'gunicorn.conf.py',
            '--bind', f'0.0.0.0:{port}',
            'app:app'
        ]
        
        print(f"🚀 Starting production server on port {port} with Gunicorn...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting Gunicorn: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("🛑 Server stopped by user")
            sys.exit(0)
    else:
        # Development: Use Flask development server
        from app import app
        
        port = int(os.environ.get('PORT', 5000))
        
        print(f"🔧 Starting development server on port {port}...")
        print("⚠️  For production, set RENDER environment variable")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            threaded=True
        )

if __name__ == "__main__":
    main()
