#!/usr/bin/env python3
"""
Script to update print statements to logging statements and remove emojis.
"""

import re
import os
from pathlib import Path

# Emoji removal patterns
emoji_patterns = [
    r'[📖🎙️📱🎬🌳📝🤖🔍🔌✅⏱️🗣️📥❌💡<�🎯📐=�>�📊🚀⚡📁🎵🧪📦🪟🎉🛠️🔧📈📱✂️🎬📝⚠️]',
    r'[🎥🎞️🔊🔉🔈📢📣📯🎺🎻🎸🥁🎧🎤🎵🎶🎼🎹]',
    r'[💻🖥️📺📷📹📽️🎦📱☎️📞📟📠]',
    r'[✨🌟⭐💫🔥💥💫🌙☀️🌞]'
]

def remove_emojis(text):
    """Remove emojis from text."""
    for pattern in emoji_patterns:
        text = re.sub(pattern, '', text)
    return text

def convert_print_to_logging(content):
    """Convert print statements to logging statements."""
    
    # Add logging import if not present
    if 'from lib.utils.logging_config import get_logger' not in content:
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if line.strip() == '' and in_imports:
                import_lines.append(line)
            elif line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
            else:
                if in_imports and line.strip():
                    import_lines.append('')
                    import_lines.append('from lib.utils.logging_config import get_logger')
                    import_lines.append('')
                    import_lines.append('logger = get_logger(__name__)')
                    import_lines.append('')
                    in_imports = False
                other_lines.append(line)
        
        content = '\n'.join(import_lines + other_lines)
    
    # Convert print statements to logging
    replacements = [
        # Error patterns
        (r'print\(f?"❌[^"]*{([^}]+)}[^"]*"\)', r'logger.error(f"\1")'),
        (r'print\("❌[^"]*"\)', r'logger.error("Error occurred")'),
        
        # Success patterns  
        (r'print\(f?"✅[^"]*{([^}]+)}[^"]*"\)', r'logger.info(f"\1")'),
        (r'print\("✅[^"]*"\)', r'logger.info("Operation successful")'),
        
        # Warning patterns
        (r'print\(f?"⚠️[^"]*{([^}]+)}[^"]*"\)', r'logger.warning(f"\1")'),
        (r'print\("⚠️[^"]*"\)', r'logger.warning("Warning")'),
        
        # Info patterns with emojis
        (r'print\(f?"[📖🎙️📱🎬🌳📝🤖🔍🔌⏱️🗣️📥💡🎯📐📊🚀⚡📁🎵🧪📦🪟🎉🛠️🔧📈✂️][^"]*{([^}]+)}[^"]*"\)', r'logger.info(f"\1")'),
        (r'print\("([📖🎙️📱🎬🌳📝🤖🔍🔌⏱️🗣️📥💡🎯📐📊🚀⚡📁🎵🧪📦🪟🎉🛠️🔧📈✂️][^"]*)"', r'logger.info("\1")'),
        
        # Debug patterns (detailed info)
        (r'print\(f?"   [^"]*{([^}]+)}[^"]*"\)', r'logger.debug(f"\1")'),
        (r'print\("   [^"]*"\)', r'logger.debug("Debug info")'),
        
        # Generic print patterns
        (r'print\(f"([^"]+)"\)', r'logger.info(f"\1")'),
        (r'print\("([^"]+)"\)', r'logger.info("\1")'),
        (r'print\(([^)]+)\)', r'logger.info(\1)'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Remove emojis
    content = remove_emojis(content)
    
    return content

def update_file(file_path):
    """Update a single file with logging."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if no print statements
        if 'print(' not in content:
            return False
        
        updated_content = convert_print_to_logging(content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Updated: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files in the project."""
    project_root = Path(__file__).parent
    lib_dir = project_root / "lib"
    
    python_files = list(lib_dir.rglob("*.py"))
    python_files.extend(project_root.glob("*.py"))
    
    updated_count = 0
    for file_path in python_files:
        if file_path.name == __file__.split('/')[-1]:  # Skip this script
            continue
        if update_file(file_path):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} files with logging")

if __name__ == "__main__":
    main()