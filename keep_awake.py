"""Prevent Windows from sleeping by periodically signaling system activity.

Uses SetThreadExecutionState to tell Windows the system is in use.
Run this in the background while long jobs are running.
Kill it (Ctrl+C or terminate process) when done.
"""
import ctypes
import time
import signal
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002  # optional: keeps display on too

def keep_awake():
    """Set execution state to prevent sleep."""
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )

def restore():
    """Restore normal sleep behavior."""
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("Sleep prevention removed. System can sleep normally.")

def signal_handler(sig, frame):
    restore()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    keep_awake()
    print("Keep-awake ACTIVE — system will not sleep.")
    print("Press Ctrl+C or kill this process to restore normal sleep.")
    try:
        while True:
            # Re-assert every 60s as a safety net
            keep_awake()
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        restore()
