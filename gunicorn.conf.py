"""Gunicorn configuration for high-throughput webhook handling."""

import os

# Worker configuration
worker_class = "gevent"
workers = int(os.environ.get("WEB_WORKERS", "4"))
worker_connections = int(os.environ.get("WORKER_CONNECTIONS", "1000"))

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Timeouts
timeout = 30
graceful_timeout = 10
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Track which worker should run the consumer
_consumer_worker_pid = None


def post_fork(server, worker):
    """Called after a worker is forked.

    Start the Huey consumer only in the first worker to avoid duplicates.
    """
    global _consumer_worker_pid

    # Only start consumer if embedded worker is enabled
    if os.environ.get("ENABLE_EMBEDDED_WORKER", "true").lower() != "true":
        return

    # Use a file lock to ensure only one worker starts the consumer
    import fcntl
    lock_file = "/tmp/huey_consumer.lock"

    try:
        lock_fd = open(lock_file, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Got the lock - this worker starts the consumer
        from agents.task_queue import start_consumer_thread

        num_workers = int(os.environ.get("QUEUE_WORKERS", "12"))
        start_consumer_thread(workers=num_workers)
        _consumer_worker_pid = worker.pid
        server.log.info(f"Huey consumer started in worker {worker.pid}")

        # Keep lock file open to maintain the lock
        worker._consumer_lock_fd = lock_fd
    except BlockingIOError:
        # Another worker has the lock - don't start consumer
        server.log.info(f"Worker {worker.pid} skipping consumer (another worker has it)")
    except Exception as e:
        server.log.error(f"Failed to start consumer in worker {worker.pid}: {e}")


def worker_exit(server, worker):
    """Called when a worker exits.

    Release the consumer lock if this worker held it.
    """
    if hasattr(worker, "_consumer_lock_fd"):
        try:
            worker._consumer_lock_fd.close()
            server.log.info(f"Released consumer lock from worker {worker.pid}")
        except Exception:
            pass
