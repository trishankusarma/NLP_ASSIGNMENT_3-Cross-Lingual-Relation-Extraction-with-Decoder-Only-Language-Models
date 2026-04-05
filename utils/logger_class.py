import sys
from datetime import datetime

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