# MoneyTree

Transform Wikipedia articles into engaging educational audio and video content with AI-powered generation and natural speech synthesis.

## Features

- **Wikipedia Integration** - Fetch comprehensive article content via REST API
- **AI Content Generation** - Create educational commentary using Ollama LLM
- **Natural TTS** - Conversational speech with ChatTTS or professional Coqui TTS
- **Video Generation** - Create vertical videos with synchronized captions for TikTok/YouTube Shorts
- **Smart Fallback** - Rule-based generation when LLM is unavailable
- **Fast Pipeline** - Wikipedia → Content Generation → Speech → Video in under 5 minutes
- **Professional Logging** - Comprehensive logging system for debugging and monitoring

## Quick Start

### Prerequisites

- Python 3.11 (required for TTS compatibility)
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) running locally (optional, for LLM generation)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd moneytree

# Install dependencies
uv add .
uv sync
```

### Basic Usage

#### Audio Generation
```bash
# Natural conversational speech (recommended)
uv run python demo_natural_tts.py "Python programming language" --engine chattts

# Professional synthetic speech
uv run python demo_coqui_tts.py "Quantum Physics" --model fast_pitch

# Compare both TTS engines
uv run python demo_natural_tts.py "Coffee" --engine both
```

#### Video Generation
```bash
# Complete pipeline: Wikipedia → Text → Speech → Video
uv run python demo_video.py "Artificial Intelligence" --template downloads/videos/minecraft_parkour.mp4 --format vertical --quality medium --engine chattts

# Fast video generation (using existing audio)
uv run python demo_fast_video.py "Cat" --quality low --preset ultrafast

# Production quality video
uv run python demo_video.py "Climate Change" --template downloads/videos/minecraft_parkour.mp4 --format vertical --quality high --engine chattts
```

## Architecture

MoneyTree follows a clean pipeline architecture:

```
Wikipedia API → Content Generator → Speech Synthesizer → Video Processor → MP4 Output
      ↓              ↓                     ↓                    ↓            ↓
   Article        Educational           Natural TTS          Captions      Vertical Video
   Content        Commentary          (ChatTTS/Coqui)       + Audio       (TikTok/YT Shorts)
```

### Components

- **Wikipedia Crawler** (`lib/wiki/`) - REST API integration with comprehensive content extraction
- **LLM Generator** (`lib/llm/llm_generator.py`) - Ollama-based content generation with WSL support
- **Rule-based Generator** (`lib/llm/discussion_generator.py`) - Fallback content creation
- **ChatTTS** (`lib/tts/chattts_speech_generator.py`) - Natural conversational speech synthesis
- **Coqui TTS** (`lib/tts/coqui_speech_generator.py`) - Professional synthetic speech synthesis
- **Video Processor** (`lib/video/clip.py`) - Video composition with synchronized captions
- **Download Manager** (`lib/download/youtube.py`) - Template video acquisition
- **Logging System** (`lib/utils/logging_config.py`) - Centralized logging configuration

## Logging System

MoneyTree includes a comprehensive logging system for debugging and monitoring:

### Log Levels

- **DEBUG** - Detailed information for diagnosing problems
- **INFO** - General information about program execution
- **WARNING** - Warning messages for unexpected situations
- **ERROR** - Error messages for failed operations
- **CRITICAL** - Critical errors that may stop execution

### Configuration

```python
from lib.utils.logging_config import setup_logging

# Basic setup (console output only)
setup_logging(log_level="INFO")

# Detailed setup with file output
setup_logging(
    log_level="DEBUG",
    log_file="logs/moneytree.log",
    console_output=True,
    detailed_format=True
)
```

### Usage in Modules

```python
from lib.utils.logging_config import get_logger, LoggedOperation

logger = get_logger(__name__)

# Basic logging
logger.info("Starting video generation")
logger.debug(f"Processing {len(segments)} caption segments")
logger.error(f"Failed to load template: {error}")

# Timed operations
with LoggedOperation(logger, "video rendering"):
    # Your code here
    pass
```

### Log Output Examples

```
INFO - moneytree.wiki.crawler - Successfully fetched summary for 'Artificial Intelligence'
DEBUG - moneytree.video.processor - Caption timing: 19 captions over 69.1s (3.6s each)
INFO - moneytree.tts.generator - ChatTTS initialized successfully
ERROR - moneytree.video.processor - Failed to load template video: File not found
```

## Audio Quality

MoneyTree generates high-quality audio with two TTS options:

### ChatTTS (Natural Speech)
- **Format**: WAV (24kHz, 16-bit)
- **Quality**: Conversational, natural-sounding
- **Voice Styles**: Natural, Expressive, Calm, Consistent
- **Best For**: Human-like narration

### Coqui TTS (Synthetic Speech)
- **Format**: WAV (22kHz, 16-bit)
- **Quality**: Professional synthetic
- **Models**: Tacotron2, FastPitch, VITS, Jenny
- **Best For**: Clear, consistent narration

## Video Generation

### Formats Supported
- **Vertical (9:16)**: TikTok, YouTube Shorts, Instagram Reels
- **Horizontal (16:9)**: Standard YouTube videos

### Quality Settings
| Quality | Resolution | FPS | Render Time | Use Case |
|---------|------------|-----|-------------|----------|
| **Low** | 540x960 | 20 | ~45s | Quick tests, drafts |
| **Medium** | 720x1280 | 24 | ~2-3min | Good balance |
| **High** | 1080x1920 | 24 | ~5-10min | Final production |

### Caption Features
- **Smart positioning**: Top third of screen (mobile-friendly)
- **Text wrapping**: Automatic line breaks to prevent cutoff
- **Synchronization**: Timed to audio duration
- **Styling**: Optimized for mobile viewing with stroke and background

## Configuration

### ChatTTS Voice Styles

| Style | Description | Use Case |
|-------|-------------|----------|
| `natural` | Balanced, natural-sounding | General use (default) |
| `expressive` | More animated and expressive | Engaging content |
| `calm` | Calm, measured delivery | Educational content |
| `consistent` | Very consistent, predictable | Professional narration |

### Video Rendering Presets

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `ultrafast` | Fastest | Basic | Quick tests |
| `superfast` | Very fast | Good | Draft videos |
| `fast` | Fast | Good | Balanced speed/quality |
| `medium` | Moderate | High | Production (default) |

### LLM Setup (Optional)

For enhanced content generation, run Ollama locally:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama3.2)
ollama pull llama3.2

# Start Ollama server
ollama serve
```

**WSL Users**: MoneyTree automatically detects WSL and connects to Windows Ollama at `172.31.32.1:11434`.

## Testing

```bash
# Run comprehensive test suite
uv run python tests/run_all_tests.py

# Test individual components
uv run python tests/wiki/test_wikipedia_crawler.py
uv run python tests/tts/test_speech_generation.py
uv run python tests/video/test_video_processing.py

# Test full pipeline integration
uv run python tests/integration/test_full_pipeline.py
```

## Performance Optimization

### Fast Video Generation
```bash
# Use existing audio (fastest)
uv run python demo_fast_video.py "Cat" --quality low --preset ultrafast

# Skip TTS generation
uv run python demo_video.py "Topic" --use-existing-audio

# Use rule-based content (no LLM)
uv run python demo_video.py "Topic" --use-rule-based
```

### Troubleshooting
- **Slow rendering**: Use `--quality low --preset ultrafast`
- **ChatTTS issues**: Numbers/dates automatically normalized
- **Font problems**: Uses system default fonts
- **Memory issues**: Template videos automatically trimmed to audio duration

## Project Structure

```
moneytree/
├── demo_video.py                    # Complete video generation pipeline
├── demo_fast_video.py              # Fast video generation (existing audio)
├── demo_natural_tts.py             # Natural TTS demo
├── demo_coqui_tts.py              # Coqui TTS demo
├── lib/
│   ├── utils/
│   │   └── logging_config.py      # Centralized logging system
│   ├── wiki/
│   │   └── crawler.py             # Wikipedia API integration
│   ├── llm/
│   │   ├── llm_generator.py       # Ollama LLM generation
│   │   └── discussion_generator.py # Rule-based fallback
│   ├── tts/
│   │   ├── chattts_speech_generator.py # Natural TTS
│   │   └── coqui_speech_generator.py   # Synthetic TTS
│   ├── video/
│   │   └── clip.py                # Video composition and rendering
│   └── download/
│       └── youtube.py             # Template video acquisition
├── tests/
│   ├── run_all_tests.py          # Comprehensive test runner
│   ├── wiki/, llm/, tts/, video/ # Component tests
│   └── integration/               # Full pipeline tests
├── audio_output/                  # Generated audio files
├── video_output/                  # Generated video files
├── downloads/                     # Template videos and music
└── logs/                         # Application logs
```

## Contributing

MoneyTree uses:
- **Google-style docstrings** for all functions and classes
- **uv** for package management
- **Professional logging** instead of print statements
- **Type hints** for better code clarity
- **Comprehensive testing** for reliability

Follow existing code patterns and ensure tests pass before submitting changes.

## License

This project is open source. See the license file for details.

## Goals

MoneyTree aims to make Wikipedia content more accessible and engaging through:

- **Educational Value**: Accurate information with engaging presentation
- **Accessibility**: Audio and video formats for different learning styles
- **Open Source**: No API keys or proprietary services required
- **Quality**: Professional-grade audio and video output
- **Flexibility**: Multiple generation methods and output formats
- **Performance**: Fast rendering optimized for content creation workflows

---

**Transform knowledge into engaging multimedia experiences with MoneyTree!**