"""
Simple test to show emojis in a pygame window.

This bypasses the full simulation and just shows emoji rendering.
"""

import sys
import pygame
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.animation.entity import RPSEntity, EntityType
from tools.animation.emoji_renderer import EmojiRenderer
from lib.utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Simple emoji rendering test."""
    print("🎮 Simple Emoji Rendering Test")
    
    # Generate emojis
    renderer = EmojiRenderer(default_size=60)
    emoji_paths = renderer.render_rps_emojis(size=60)
    
    if not emoji_paths:
        print("❌ Failed to generate emojis")
        return False
    
    print(f"✅ Generated emojis: {list(emoji_paths.keys())}")
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SRCALPHA)
    pygame.display.set_caption("Emoji Test")
    clock = pygame.time.Clock()
    
    # Create entities
    entities = []
    positions = [(200, 200), (400, 200), (600, 200)]
    entity_types = [EntityType.ROCK, EntityType.PAPER, EntityType.SCISSORS]
    
    for i, (pos, entity_type) in enumerate(zip(positions, entity_types)):
        entity = RPSEntity(
            entity_type=entity_type,
            position=pos,
            velocity=(0, 0),
            radius=40
        )
        
        # Set emoji
        emoji_path = emoji_paths.get(entity_type.value)
        if emoji_path:
            entity.emoji_image_path = emoji_path
            entity.use_emoji = True
            print(f"✅ Set emoji for {entity_type.value}: {emoji_path}")
        
        entities.append(entity)
    
    # Add labels
    font = pygame.font.Font(None, 36)
    labels = [
        font.render("Rock 🪨", True, (255, 255, 255)),
        font.render("Paper 📄", True, (255, 255, 255)),
        font.render("Scissors ✂️", True, (255, 255, 255))
    ]
    
    # Main loop
    running = True
    frame_count = 0
    
    print("🚀 Starting visual test...")
    print("You should see emojis instead of colored circles")
    print("Press ESC or close window to exit")
    
    while running and frame_count < 300:  # 5 seconds at 60 FPS
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill((40, 40, 40))  # Dark gray background
        
        # Render entities
        for entity in entities:
            entity.render(screen)
        
        # Render labels
        for i, label in enumerate(labels):
            screen.blit(label, (positions[i][0] - label.get_width() // 2, positions[i][1] + 60))
        
        # Add instructions
        instruction = font.render("Press ESC to exit", True, (255, 255, 255))
        screen.blit(instruction, (10, 10))
        
        # Add frame counter
        frame_text = font.render(f"Frame: {frame_count}", True, (255, 255, 255))
        screen.blit(frame_text, (10, 50))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
    
    pygame.quit()
    print(f"✅ Rendered {frame_count} frames")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Simple emoji test completed!")
    else:
        print("\n⚠️ Simple emoji test failed.")