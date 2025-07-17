"""
Debug script to check emoji loading and rendering.

This script helps identify where the emoji loading process is failing.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.animation.entity import RPSEntity, EntityType
from tools.animation.emoji_renderer import EmojiRenderer
from lib.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import pygame
    from PIL import Image
    HAS_PYGAME = True
    HAS_PIL = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_PYGAME = False
    HAS_PIL = False


def test_emoji_files():
    """Test if emoji files are valid and can be loaded."""
    print("🔍 Testing emoji file generation...")
    
    renderer = EmojiRenderer(default_size=40)
    emoji_paths = renderer.render_rps_emojis(size=40)
    
    print(f"Generated emoji paths: {emoji_paths}")
    
    for entity_type, path in emoji_paths.items():
        print(f"\n🔍 Testing {entity_type} emoji: {path}")
        
        # Check if file exists
        if not os.path.exists(path):
            print(f"❌ File does not exist: {path}")
            continue
        
        # Check file size
        file_size = os.path.getsize(path)
        print(f"✅ File exists, size: {file_size} bytes")
        
        # Try to load with PIL
        try:
            with Image.open(path) as img:
                print(f"✅ PIL can load image: {img.size}, mode: {img.mode}")
        except Exception as e:
            print(f"❌ PIL failed to load: {e}")
            continue
        
        # Try to load with pygame (if available)
        if HAS_PYGAME:
            try:
                # Initialize pygame for testing
                pygame.init()
                pygame.display.set_mode((100, 100), pygame.SRCALPHA)
                
                # Try to load with pygame
                surface = pygame.image.load(path)
                print(f"✅ Pygame can load image: {surface.get_size()}")
                
                # Try to convert with alpha
                surface_alpha = surface.convert_alpha()
                print(f"✅ Pygame alpha conversion successful")
                
                pygame.quit()
                
            except Exception as e:
                print(f"❌ Pygame failed to load: {e}")
                pygame.quit()
                continue
    
    return emoji_paths


def test_entity_loading():
    """Test if entities can load emoji surfaces."""
    print("\n🔍 Testing entity emoji loading...")
    
    # Generate emojis
    emoji_paths = test_emoji_files()
    
    if not emoji_paths:
        print("❌ No emoji paths available for testing")
        return False
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SRCALPHA)
    pygame.display.set_caption("Emoji Loading Test")
    
    print("✅ Pygame initialized")
    
    # Create test entity
    entity = RPSEntity.create_random(
        screen_width=800,
        screen_height=600,
        entity_type=EntityType.ROCK
    )
    
    print(f"✅ Entity created: {entity.entity_type}")
    
    # Test emoji loading
    rock_path = emoji_paths.get('rock')
    if rock_path:
        print(f"🔍 Testing emoji loading for path: {rock_path}")
        
        # Set emoji path
        entity.emoji_image_path = rock_path
        entity.use_emoji = True
        
        print(f"✅ Set emoji path: {entity.emoji_image_path}")
        print(f"✅ Use emoji: {entity.use_emoji}")
        
        # Try to load emoji surface
        entity._load_emoji_surface()
        
        if entity._emoji_surface:
            print(f"✅ Emoji surface loaded successfully: {entity._emoji_surface.get_size()}")
        else:
            print(f"❌ Emoji surface failed to load")
            
            # Try manual loading to debug
            print("🔍 Trying manual loading...")
            try:
                pil_image = Image.open(rock_path)
                print(f"✅ PIL loaded: {pil_image.size}, mode: {pil_image.mode}")
                
                if pil_image.mode != 'RGBA':
                    pil_image = pil_image.convert('RGBA')
                    print(f"✅ Converted to RGBA")
                
                emoji_size = int(entity.radius * 1.8)
                pil_image = pil_image.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
                print(f"✅ Resized to {emoji_size}x{emoji_size}")
                
                data = pil_image.tobytes()
                print(f"✅ Got image data: {len(data)} bytes")
                
                surface = pygame.image.fromstring(data, pil_image.size, pil_image.mode)
                print(f"✅ Created pygame surface: {surface.get_size()}")
                
                surface_alpha = surface.convert_alpha()
                print(f"✅ Converted to alpha: {surface_alpha.get_size()}")
                
            except Exception as e:
                print(f"❌ Manual loading failed: {e}")
                import traceback
                traceback.print_exc()
    
    pygame.quit()
    return True


def test_rendering():
    """Test actual rendering of emojis."""
    print("\n🔍 Testing emoji rendering...")
    
    # Generate emojis
    renderer = EmojiRenderer(default_size=40)
    emoji_paths = renderer.render_rps_emojis(size=40)
    
    if not emoji_paths:
        print("❌ No emoji paths available")
        return False
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SRCALPHA)
    pygame.display.set_caption("Emoji Rendering Test")
    clock = pygame.time.Clock()
    
    # Create entities with emojis
    entities = []
    for i, (entity_type_str, emoji_path) in enumerate(emoji_paths.items()):
        entity_type = EntityType(entity_type_str)
        entity = RPSEntity(
            entity_type=entity_type,
            position=(200 + i * 200, 300),
            velocity=(0, 0),
            radius=30
        )
        
        entity.emoji_image_path = emoji_path
        entity.use_emoji = True
        
        entities.append(entity)
        print(f"✅ Created {entity_type_str} entity with emoji: {emoji_path}")
    
    # Render loop
    running = True
    frames = 0
    
    while running and frames < 60:  # Run for 60 frames (~1 second)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Clear screen
        screen.fill((50, 50, 50, 255))  # Dark gray background
        
        # Render entities
        for entity in entities:
            entity.render(screen)
        
        # Add debug info
        font = pygame.font.Font(None, 24)
        text = font.render(f"Frame {frames}: Testing emoji rendering", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)
        frames += 1
    
    pygame.quit()
    print(f"✅ Rendered {frames} frames")
    return True


def main():
    """Run all emoji debugging tests."""
    print("🐛 Emoji Loading Debug Suite")
    print("=" * 50)
    
    if not (HAS_PYGAME and HAS_PIL):
        print("❌ Missing dependencies:")
        print(f"   Pygame: {HAS_PYGAME}")
        print(f"   PIL: {HAS_PIL}")
        return False
    
    try:
        # Test 1: File generation and loading
        print("\n1️⃣ Testing file generation...")
        emoji_paths = test_emoji_files()
        
        if not emoji_paths:
            print("❌ Failed to generate emoji files")
            return False
        
        # Test 2: Entity loading
        print("\n2️⃣ Testing entity loading...")
        if not test_entity_loading():
            print("❌ Failed entity loading test")
            return False
        
        # Test 3: Rendering
        print("\n3️⃣ Testing rendering...")
        if not test_rendering():
            print("❌ Failed rendering test")
            return False
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Debug suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Emoji system is working correctly!")
    else:
        print("\n⚠️ Emoji system has issues that need fixing.")