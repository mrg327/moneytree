"""
Comprehensive diagnostic tool to identify why emojis are not displaying.

This tool tests every step of the emoji rendering pipeline to identify the failure point.
"""

import sys
import os
import pygame
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


def test_emoji_generation():
    """Test emoji file generation step by step."""
    print("🔍 Step 1: Testing emoji file generation...")
    
    renderer = EmojiRenderer(default_size=40)
    
    # Test font loading
    print(f"Available emoji fonts: {len(renderer.emoji_fonts)}")
    for i, font_path in enumerate(renderer.emoji_fonts):
        print(f"  {i+1}. {font_path}")
    
    if not renderer.emoji_fonts:
        print("❌ CRITICAL: No emoji fonts found!")
        return False
    
    # Test emoji generation
    emoji_paths = renderer.render_rps_emojis(size=40)
    print(f"Generated emoji paths: {emoji_paths}")
    
    for entity_type, path in emoji_paths.items():
        if not os.path.exists(path):
            print(f"❌ CRITICAL: Emoji file missing: {path}")
            return False
        
        file_size = os.path.getsize(path)
        print(f"✅ {entity_type}: {path} ({file_size} bytes)")
    
    return emoji_paths


def test_emoji_font_rendering():
    """Test if the emoji fonts can actually render emojis."""
    print("\n🔍 Step 2: Testing emoji font rendering capabilities...")
    
    renderer = EmojiRenderer(default_size=40)
    
    # Test each font individually
    test_emojis = ['🪨', '📄', '✂️', '😊']
    
    for font_path in renderer.emoji_fonts:
        print(f"\n📝 Testing font: {font_path}")
        
        try:
            from PIL import ImageFont, ImageDraw, Image
            
            # Try to load font
            font = ImageFont.truetype(font_path, 40)
            print(f"✅ Font loaded successfully")
            
            # Test rendering each emoji
            for emoji in test_emojis:
                try:
                    # Create test image
                    img = Image.new('RGBA', (50, 50), (255, 255, 255, 0))
                    draw = ImageDraw.Draw(img)
                    
                    # Get text bbox
                    bbox = draw.textbbox((0, 0), emoji, font=font)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    print(f"  {emoji}: bbox={bbox}, size={width}x{height}")
                    
                    if width > 5 and height > 5:
                        print(f"  ✅ {emoji} renders properly")
                    else:
                        print(f"  ⚠️ {emoji} may not render (too small)")
                        
                except Exception as e:
                    print(f"  ❌ {emoji} failed: {e}")
                    
        except Exception as e:
            print(f"❌ Font failed to load: {e}")
    
    return True


def test_pygame_emoji_loading():
    """Test if pygame can load the generated emoji images."""
    print("\n🔍 Step 3: Testing pygame emoji loading...")
    
    # Generate emojis first
    renderer = EmojiRenderer(default_size=40)
    emoji_paths = renderer.render_rps_emojis(size=40)
    
    if not emoji_paths:
        print("❌ CRITICAL: No emoji paths to test")
        return False
    
    # Initialize pygame
    pygame.init()
    pygame.display.set_mode((100, 100), pygame.SRCALPHA)
    
    results = {}
    
    for entity_type, path in emoji_paths.items():
        print(f"\n📝 Testing {entity_type}: {path}")
        
        try:
            # Test pygame image loading
            surface = pygame.image.load(path)
            print(f"✅ Pygame loaded: {surface.get_size()}")
            
            # Test alpha conversion
            surface_alpha = surface.convert_alpha()
            print(f"✅ Alpha conversion: {surface_alpha.get_size()}")
            
            # Test PIL to pygame conversion (the method we use)
            pil_image = Image.open(path)
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
            
            # Scale like entity does
            emoji_size = int(20 * 1.8)  # Default entity radius * 1.8
            pil_image = pil_image.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
            
            # Convert to pygame surface
            data = pil_image.tobytes()
            manual_surface = pygame.image.fromstring(data, pil_image.size, pil_image.mode)
            manual_surface = manual_surface.convert_alpha()
            
            print(f"✅ Manual conversion: {manual_surface.get_size()}")
            
            results[entity_type] = {
                'path': path,
                'surface': manual_surface,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ {entity_type} failed: {e}")
            results[entity_type] = {
                'path': path,
                'surface': None,
                'success': False
            }
    
    pygame.quit()
    return results


def test_entity_emoji_integration():
    """Test if entities properly integrate with emoji rendering."""
    print("\n🔍 Step 4: Testing entity emoji integration...")
    
    # Generate emojis
    renderer = EmojiRenderer(default_size=40)
    emoji_paths = renderer.render_rps_emojis(size=40)
    
    if not emoji_paths:
        print("❌ CRITICAL: No emoji paths available")
        return False
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 200), pygame.SRCALPHA)
    
    # Create test entities
    entities = []
    entity_types = [EntityType.ROCK, EntityType.PAPER, EntityType.SCISSORS]
    
    for i, entity_type in enumerate(entity_types):
        entity = RPSEntity(
            entity_type=entity_type,
            position=(100 + i * 100, 100),
            velocity=(0, 0),
            radius=30
        )
        
        # Set emoji
        emoji_path = emoji_paths.get(entity_type.value)
        if emoji_path:
            entity.emoji_image_path = emoji_path
            entity.use_emoji = True
            entity.set_emoji_path_mapping(emoji_paths)
            
            print(f"✅ Entity {entity_type.value}: emoji_path={emoji_path}")
            print(f"   use_emoji={entity.use_emoji}")
            
            # Force load emoji surface
            entity._load_emoji_surface()
            
            if entity._emoji_surface:
                print(f"   ✅ Emoji surface loaded: {entity._emoji_surface.get_size()}")
            else:
                print(f"   ❌ Emoji surface failed to load")
        else:
            print(f"❌ No emoji path for {entity_type.value}")
        
        entities.append(entity)
    
    # Test rendering
    print(f"\n📝 Testing rendering...")
    
    for frame in range(3):  # Test a few frames
        screen.fill((50, 50, 50))  # Dark background
        
        for entity in entities:
            # Check rendering path
            if entity.use_emoji and entity._emoji_surface:
                print(f"Frame {frame}: {entity.entity_type.value} using emoji rendering")
            else:
                print(f"Frame {frame}: {entity.entity_type.value} using circle fallback")
            
            entity.render(screen)
        
        pygame.display.flip()
        pygame.time.wait(100)
    
    # Save a test image
    pygame.image.save(screen, "emoji_integration_test.png")
    print(f"✅ Test image saved to: emoji_integration_test.png")
    
    pygame.quit()
    return True


def test_font_availability():
    """Test what fonts are actually available on the system."""
    print("\n🔍 Step 5: Testing system font availability...")
    
    font_candidates = [
        # Windows emoji fonts (accessible via WSL2)
        "/mnt/c/Windows/Fonts/seguiemj.ttf",
        "/mnt/c/Windows/Fonts/seguisym.ttf",
        "/mnt/c/Windows/Fonts/segmdl2.ttf",
        
        # Linux emoji fonts
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
        "/usr/share/fonts/truetype/color-emoji/NotoColorEmoji.ttf",
        
        # Fallback fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
    ]
    
    available_fonts = []
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                from PIL import ImageFont
                font = ImageFont.truetype(font_path, 24)
                available_fonts.append(font_path)
                print(f"✅ Available: {font_path}")
            except Exception as e:
                print(f"⚠️ Exists but can't load: {font_path} - {e}")
        else:
            print(f"❌ Missing: {font_path}")
    
    print(f"\n📊 Summary: {len(available_fonts)} fonts available out of {len(font_candidates)} candidates")
    
    return available_fonts


def main():
    """Run comprehensive emoji diagnostics."""
    print("🐛 Comprehensive Emoji Diagnostic Suite")
    print("=" * 60)
    
    if not (HAS_PYGAME and HAS_PIL):
        print("❌ Missing dependencies:")
        print(f"   Pygame: {HAS_PYGAME}")
        print(f"   PIL: {HAS_PIL}")
        return False
    
    try:
        # Test 1: Font availability
        available_fonts = test_font_availability()
        if not available_fonts:
            print("❌ CRITICAL: No fonts available")
            return False
        
        # Test 2: Font rendering capabilities
        test_emoji_font_rendering()
        
        # Test 3: Emoji file generation
        emoji_paths = test_emoji_generation()
        if not emoji_paths:
            print("❌ CRITICAL: Emoji generation failed")
            return False
        
        # Test 4: Pygame loading
        pygame_results = test_pygame_emoji_loading()
        if not pygame_results:
            print("❌ CRITICAL: Pygame loading failed")
            return False
        
        # Test 5: Entity integration
        entity_results = test_entity_emoji_integration()
        if not entity_results:
            print("❌ CRITICAL: Entity integration failed")
            return False
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        print(f"✅ System fonts available: {len(available_fonts)}")
        print(f"✅ Emoji files generated: {len(emoji_paths)}")
        print(f"✅ Pygame loading: {len([r for r in pygame_results.values() if r['success']])}/{len(pygame_results)}")
        print(f"✅ Entity integration: Complete")
        
        # If everything passes but emojis still don't show, the issue is elsewhere
        print("\n🔍 CONCLUSION:")
        print("All diagnostic tests passed, but emojis still don't display.")
        print("This suggests the issue is in the rendering logic or timing.")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Diagnostics completed - check output for specific issues")
    else:
        print("\n⚠️ Diagnostics failed - fundamental issues detected")