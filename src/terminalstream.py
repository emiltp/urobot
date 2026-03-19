import logging
import os
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor

_log = logging.getLogger(__name__)

class TerminalStream(QObject):
    """Custom stream handler that redirects stdout/stderr to a QTextEdit widget and log file."""
    text_written = pyqtSignal(str)
    
    def __init__(self, text_widget, log_file_path=None):
        super().__init__()
        self.text_widget = text_widget
        self.log_file_path = log_file_path
        self.log_file = None
        self.text_written.connect(self.append_text)
        
        # Open log file if path is provided
        if self.log_file_path:
            try:
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                # Open in append mode with line buffering
                self.log_file = open(self.log_file_path, 'a', encoding='utf-8', buffering=1)
                # Write session header if file is empty
                if self.log_file.tell() == 0:
                    self.log_file.write(f"{'='*80}\n")
                    self.log_file.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    self.log_file.write(f"{'='*80}\n")
                    self.log_file.flush()
            except Exception as e:
                print(f"Warning: Could not open log file {self.log_file_path}: {e}")
                self.log_file = None
    
    def write(self, text):
        # Emit all text (including newlines and empty strings)
        if not text:
            return
        self.text_written.emit(text)
        
        # Also write to log file immediately
        if self.log_file:
            try:
                timestamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ')
                # Write only non-empty lines, one per line
                for line in text.split('\n'):
                    if line.strip():
                        self.log_file.write(timestamp + line + '\n')
                self.log_file.flush()
            except Exception as e:
                _log.warning("TerminalStream log file write failed: %s", e)
    
    def append_text(self, text):
        """Append text to the console widget (called in main thread)."""
        cursor = self.text_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.text_widget.setTextCursor(cursor)
        self.text_widget.ensureCursorVisible()
    
    def flush(self):
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception as e:
                _log.warning("TerminalStream log file flush failed: %s", e)
    
    def __del__(self):
        """Ensure log file is closed on cleanup (e.g. crash)."""
        self.close()

    def close(self):
        """Close the log file and write session end marker."""
        if self.log_file:
            try:
                self.log_file.write(f"{'='*80}\n")
                self.log_file.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write(f"{'='*80}\n")
                self.log_file.flush()
                self.log_file.close()
            except Exception as e:
                _log.warning("TerminalStream log file close failed: %s", e)
            self.log_file = None