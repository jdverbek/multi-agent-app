#!/usr/bin/env python3
"""
Gunicorn configuration for multi-agent app with proper timeout handling
"""

import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings - Critical for long-running tasks
timeout = 60  # Worker timeout (60 seconds for API responses)
keepalive = 5  # Keep-alive timeout
graceful_timeout = 60  # Graceful shutdown timeout

# Memory management
max_worker_memory = 512 * 1024 * 1024  # 512MB per worker
preload_app = False  # Don't preload to avoid memory issues

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "multi-agent-app"

# Security
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 8192

# Performance
enable_stdio_inheritance = True
reuse_port = True

def when_ready(server):
    """Called when the server is ready to accept connections."""
    server.log.info("Multi-agent app server is ready to accept connections")

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called before a worker is forked."""
    server.log.info(f"Worker {worker.pid} is about to be forked")

def post_fork(server, worker):
    """Called after a worker is forked."""
    server.log.info(f"Worker {worker.pid} has been forked")

def worker_abort(worker):
    """Called when a worker is aborted."""
    worker.log.error(f"Worker {worker.pid} was aborted - likely due to timeout or memory issues")

