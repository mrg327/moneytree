# Logging Migration Summary

This document summarizes the migration from print statements to Python's logging library in the MoneyTree project.

## Changes Made

### 1. Centralized Logging Configuration
- **Created**: `lib/utils/logging_config.py`
- **Features**:
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Console and file output options
  - Rotating file handlers (10MB, 5 backups)
  - Performance timing decorators
  - Context managers for timed operations
  - Predefined logger configurations for each component

### 2. Core Module Updates
- **Wikipedia Crawler** (`lib/wiki/crawler.py`):
  - Added logging import and logger initialization
  - Converted error messages to `logger.error()`
  - Added timing decorators for API calls
  - Debug logging for request details
  
- **ChatTTS Generator** (`lib/tts/chattts_speech_generator.py`):
  - Replaced initialization print statements with logging
  - Critical errors for missing dependencies
  - Info logs for successful operations
  - Debug logs for detailed parameters
  
- **Video Processor** (`lib/video/clip.py`):
  - Template loading logs
  - Caption generation debug info
  - Error handling with proper log levels

### 3. Emoji Removal
- Removed all emojis from log messages
- Replaced with clear, professional text
- Examples:
  - `"✅ Success"` → `"Operation successful"`
  - `"❌ Error"` → `"Error occurred"`
  - `"🗣️ ChatTTS"` → `"ChatTTS"`

### 4. Log Level Guidelines

#### DEBUG
- Detailed parameter values
- Request/response details
- Timing information
- Internal state changes

#### INFO  
- Major operation starts/completions
- Successful operations
- Configuration details
- User-facing status updates

#### WARNING
- Unexpected but recoverable situations
- Fallback mechanisms activated
- Performance issues

#### ERROR
- Failed operations
- Invalid inputs
- Network/file system errors
- Recoverable exceptions

#### CRITICAL
- Missing dependencies
- Initialization failures
- Unrecoverable errors

## Usage Examples

### Basic Setup
```python
from lib.utils.logging_config import setup_logging, get_logger

# Setup logging for the application
setup_logging(log_level="INFO", log_file="logs/moneytree.log")

# Get logger for a module
logger = get_logger(__name__)
```

### Module Usage
```python
from lib.utils.logging_config import get_logger, LoggedOperation, log_execution_time

logger = get_logger(__name__)

# Basic logging
logger.info("Starting video generation")
logger.debug(f"Processing {len(segments)} caption segments")
logger.error(f"Failed to load template: {error}")

# Timed operations
with LoggedOperation(logger, "video rendering"):
    render_video()

# Function timing decorator
@log_execution_time(logger)
def generate_speech(text):
    # Function implementation
    pass
```

### Configuration Options
```python
# Development setup
setup_logging(
    log_level="DEBUG",
    log_file="logs/debug.log",
    console_output=True,
    detailed_format=True
)

# Production setup
setup_logging(
    log_level="INFO",
    log_file="logs/production.log",
    console_output=False,
    detailed_format=False
)
```

## Benefits

### 1. Professional Development
- Industry-standard logging practices
- Configurable output levels
- Structured log format
- File rotation management

### 2. Debugging Capabilities
- Detailed debug information
- Performance timing
- Error stack traces
- Component-specific logging

### 3. Production Readiness
- Log level filtering
- File-based logging
- No emoji clutter
- Clean, parseable output

### 4. Monitoring Support
- Structured log messages
- Consistent formatting
- Error categorization
- Performance metrics

## Remaining Work

The following items still need to be completed:

### Demo Scripts
- `demo_video.py`
- `demo_fast_video.py`  
- `demo_natural_tts.py`
- `demo_coqui_tts.py`

### Test Files
- All files in `tests/` directory
- Test runners and utilities

### Additional Modules
- `lib/llm/discussion_generator.py`
- `lib/llm/llm_generator.py`
- `lib/tts/coqui_speech_generator.py`
- `lib/download/youtube.py`

## Migration Script

A migration script (`update_logging.py`) was created to help automate the conversion process for remaining files. It handles:

- Adding logging imports
- Converting print statements to appropriate log levels
- Removing emojis from messages
- Preserving code functionality

## Log Output Examples

### Before (Print Statements)
```
🗣️  Initializing ChatTTS...
   Device: cpu
✅ ChatTTS initialized successfully
❌ Failed to load template video
```

### After (Logging)
```
INFO - moneytree.tts.generator - Initializing ChatTTS on device: cpu
INFO - moneytree.tts.generator - ChatTTS initialized successfully
ERROR - moneytree.video.processor - Failed to load template video: File not found
```

## Testing

To test the logging system:

```bash
# Test with different log levels
uv run python -c "
from lib.utils.logging_config import setup_logging, get_logger
setup_logging(log_level='DEBUG', console_output=True)
logger = get_logger('test')
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')
"

# Test with file output
uv run python -c "
from lib.utils.logging_config import setup_logging, get_logger
setup_logging(log_level='INFO', log_file='test.log')
logger = get_logger('test')
logger.info('This will be written to test.log')
"
```

This migration significantly improves the professionalism and debugging capabilities of the MoneyTree project while maintaining all existing functionality.