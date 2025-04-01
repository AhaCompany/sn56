import time
import threading
from fiber.logging_utils import get_logger


logger = get_logger(__name__)


def stream_logs(container, timeout=3600, max_log_size=1000000):
    """
    Stream logs from a Docker container with safety features.
    
    Args:
        container: Docker container object
        timeout: Maximum time to wait for logs in seconds (default: 1 hour)
        max_log_size: Maximum log buffer size in characters (default: 1MB)
        
    Returns:
        str: The complete log output
    """
    buffer = ""
    complete_logs = ""
    start_time = time.time()
    log_done = threading.Event()
    
    def log_timeout_handler():
        """Thread to stop log streaming if it takes too long"""
        timeout_wait = min(timeout, 3600)  # Cap at 1 hour max
        time_waited = 0
        while not log_done.is_set() and time_waited < timeout_wait:
            time.sleep(5)  # Check every 5 seconds
            time_waited += 5
            
        if not log_done.is_set():
            logger.warning(f"Log streaming timed out after {timeout_wait} seconds")
            log_done.set()  # Signal main thread to stop
    
    # Start timeout handler thread
    timeout_thread = threading.Thread(target=log_timeout_handler, daemon=True)
    timeout_thread.start()
    
    try:
        # Use a generator with timeout to get logs incrementally
        log_generator = container.logs(stream=True, follow=True)
        
        while not log_done.is_set():
            try:
                # Use a non-blocking approach with timeout
                log_chunk = next(log_generator, None)
                if log_chunk is None:
                    # End of logs
                    break
                    
                log_text = log_chunk.decode("utf-8", errors="replace")
                buffer += log_text
                complete_logs += log_text
                
                # Prevent buffer from growing too large
                if len(complete_logs) > max_log_size:
                    logger.warning(f"Log size exceeded {max_log_size} characters, truncating")
                    complete_logs = complete_logs[-max_log_size:]
                
                # Process buffer by lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        logger.info(line)
                        
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Log streaming reached timeout of {timeout} seconds")
                    break
                    
            except StopIteration:
                # End of logs
                break
                
            except Exception as chunk_error:
                logger.error(f"Error processing log chunk: {str(chunk_error)}")
                # Continue trying to get more logs
                time.sleep(1)
                
        # Log any remaining partial lines
        if buffer:
            logger.info(buffer)
            
    except Exception as e:
        logger.error(f"Error streaming logs: {str(e)}")
        
    finally:
        # Signal timeout thread to exit
        log_done.set()
        
        # Try to get any remaining logs
        try:
            remaining_logs = container.logs(tail=100, stream=False).decode("utf-8", errors="replace")
            if remaining_logs and remaining_logs not in complete_logs:
                logger.info("Captured remaining logs:")
                for line in remaining_logs.splitlines():
                    if line:
                        logger.info(line)
                complete_logs += remaining_logs
        except Exception as tail_error:
            logger.error(f"Error getting remaining logs: {str(tail_error)}")
            
    return complete_logs
