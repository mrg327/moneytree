"""
Simple test to verify emoji rendering fix.
"""

import sys
import pygame
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.animation.entity import RPSEntity, EntityType
from tools.animation.emoji_renderer import EmojiRenderer


def main():
    """Test emoji rendering fix."""
    print("🔍 Testing emoji rendering fix...")
    
    # Generate emojis
    renderer = EmojiRenderer(default_size=40)
    emoji_paths = renderer.render_rps_emojis(size=40)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 200), pygame.SRCALPHA)
    pygame.display.set_caption("Emoji Fix Test")
    
    # Create entities
    entities = []
    entity_types = [EntityType.ROCK, EntityType.PAPER, EntityType.SCISSORS]
    
    for i, entity_type in enumerate(entity_types):
        entity = RPSEntity(
            entity_type=entity_type,
            position=(150 + i * 150, 100),
            velocity=(0, 0),
            radius=30
        )
        
        # Set emoji
        emoji_path = emoji_paths.get(entity_type.value)
        if emoji_path:
            entity.emoji_image_path = emoji_path
            entity.use_emoji = True
            entity.set_emoji_path_mapping(emoji_paths)
            print(f"✅ Entity {entity_type.value}: use_emoji={entity.use_emoji}, path={emoji_path}")
        
        entities.append(entity)
    
    # Test rendering
    print("🎨 Testing rendering...")
    
    clock = pygame.time.Clock()
    frame_count = 0
    
    while frame_count < 90:  # 3 seconds at 30 FPS
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Render entities
        for entity in entities:
            entity.render(screen)
        
        # Show frame counter
        font = pygame.font.Font(None, 36)
        text = font.render(f"Frame {frame_count}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1
    
    # Save final frame
    pygame.image.save(screen, "emoji_fix_test.png")
    print("✅ Saved test image: emoji_fix_test.png")
    
    pygame.quit()
    print(f"✅ Completed {frame_count} frames")
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Emoji fix test completed!")
        print("Check emoji_fix_test.png to see if emojis are displaying")
    else:
        print("\n❌ Emoji fix test failed")