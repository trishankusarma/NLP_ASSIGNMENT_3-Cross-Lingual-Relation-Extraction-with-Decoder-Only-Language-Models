import sys
import os
from datetime import datetime

def logging(s='1'):
    global current_logger
    log_path = os.path.join('logs', f'output_{s}.txt')

    # If stdout is already a Logger, unwrap to real stdout
    if isinstance(sys.stdout, Logger):
        sys.stdout = sys.stdout.terminal
    
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f'output_{s}.txt')

    # Initialize new logger (always starts fresh)
    current_logger = Logger(log_path)
    sys.stdout = current_logger

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"Logging started at {timestamp}")
    print(f"Log file   : {log_path}")
    print("=" * 70)

class Logger:
    def __init__(self, filename):
        # Unwrap if already a Logger to avoid double-wrapping
        if isinstance(sys.stdout, Logger):
            self.terminal = sys.stdout.terminal
        else:
            self.terminal = sys.stdout
        
        self.log = open(filename, "w", buffering=1, encoding='utf-8')
        self._buffer = ""  # accumulate partial lines

    def write(self, message):
        # Always write to terminal as-is (tqdm needs raw control chars)
        self.terminal.write(message)
        self.terminal.flush()

        # For log file: skip tqdm animation lines (contain \r but no \n)
        # Only log complete lines (ending with \n)
        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            line = line.replace('\r', '').strip()
            if line:  # skip empty lines
                timestamp = datetime.now().strftime("[%H:%M:%S] ")
                self.log.write(timestamp + line + '\n')
                self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        # Critical: tqdm checks this to decide whether to show progress bar
        return self.terminal.isatty()