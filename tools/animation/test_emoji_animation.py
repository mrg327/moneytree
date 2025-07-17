"""
Test script to demonstrate emoji rendering in RPS animation.

This script creates a simple RPS simulation with emoji rendering enabled,
showing how the emoji images are overlaid on the entities.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.animation.rps_simulator import RPSSimulator, SimulationConfig
from lib.animation.animation_engine import AnimationEngine, AnimationConfig
from lib.animation.entity import RPSEntity, EntityType
from tools.animation.emoji_renderer import EmojiRenderer
from lib.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_emoji_animation():
    """Test RPS animation with emoji rendering."""
    print("🎮 Testing RPS Animation with Emoji Rendering...")
    
    # Create emoji renderer and generate RPS emojis
    emoji_renderer = EmojiRenderer(default_size=40)  # Smaller size for animation
    emoji_paths = emoji_renderer.render_rps_emojis(size=40)
    
    if not emoji_paths:
        print("❌ Failed to render emojis - falling back to circle rendering")
        return False
    
    print(f"✅ Generated emojis: {list(emoji_paths.keys())}")
    
    # Create animation configuration
    config = AnimationConfig(
        screen_width=800,
        screen_height=600,
        fps=60,
        headless=False,   # Show the animation window
        export_frames=False,  # Don't export frames for this test
        show_fps=True,
        show_stats=True
    )
    
    # Create simulation config
    sim_config = SimulationConfig(
        screen_width=800,
        screen_height=600,
        total_entities=30,
        entity_distribution={
            EntityType.ROCK: 0.33,
            EntityType.PAPER: 0.33,
            EntityType.SCISSORS: 0.34
        }
    )
    
    # Create RPS simulator
    simulator = RPSSimulator(sim_config)
    
    # Create entities with emoji rendering
    entities = []
    for i in range(sim_config.total_entities):
        # Determine entity type based on ratios
        if i < sim_config.total_entities * sim_config.entity_distribution[EntityType.ROCK]:
            entity_type = EntityType.ROCK
        elif i < sim_config.total_entities * (sim_config.entity_distribution[EntityType.ROCK] + sim_config.entity_distribution[EntityType.PAPER]):
            entity_type = EntityType.PAPER
        else:
            entity_type = EntityType.SCISSORS
        
        # Create entity
        entity = RPSEntity.create_random(
            screen_width=config.screen_width,
            screen_height=config.screen_height,
            entity_type=entity_type
        )
        
        # Store emoji path for later loading (after pygame init)
        emoji_path = emoji_paths.get(entity_type.value)
        if emoji_path:
            entity.emoji_image_path = emoji_path
            entity.use_emoji = True
            print(f"✅ Set emoji path for {entity_type.value}: {emoji_path}")
        else:
            print(f"⚠️ No emoji found for {entity_type.value}, using circle rendering")
        
        entities.append(entity)
    
    # Set entities in simulator
    simulator.entities = entities
    
    # Create animation engine
    engine = AnimationEngine(config, sim_config)
    
    try:
        print("🚀 Starting emoji animation...")
        print("Press ESC or close window to stop")
        
        # Set the simulator in the engine
        engine.simulator = simulator
        
        # Run the simulation
        results = engine.run_simulation()
        
        print("\n🏆 Animation Results:")
        print(f"Duration: {results.get('duration', 0):.2f} seconds")
        print(f"Total battles: {results.get('total_battles', 0)}")
        print(f"Winner: {results.get('winner', 'None')}")
        print(f"Final counts: {results.get('final_counts', {})}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Animation stopped by user")
        return True
    except Exception as e:
        print(f"❌ Animation failed: {e}")
        return False


def test_emoji_fallback():
    """Test that animation works with fallback to circles when emojis fail."""
    print("\n🔄 Testing fallback to circle rendering...")
    
    # Create animation configuration
    config = AnimationConfig(
        screen_width=800,
        screen_height=600,
        fps=60,
        headless=False,
        export_frames=False,
        show_fps=True,
        show_stats=True
    )
    
    # Create simple simulation
    sim_config = SimulationConfig(
        screen_width=800,
        screen_height=600,
        total_entities=15
    )
    simulator = RPSSimulator(sim_config)
    
    # Create entities WITHOUT emoji rendering (should use circles)
    entities = []
    for i in range(sim_config.total_entities):
        entity_type = [EntityType.ROCK, EntityType.PAPER, EntityType.SCISSORS][i % 3]
        entity = RPSEntity.create_random(
            screen_width=config.screen_width,
            screen_height=config.screen_height,
            entity_type=entity_type
        )
        # Note: NOT setting emoji, so should use circle rendering
        entities.append(entity)
    
    simulator.entities = entities
    
    # Run animation
    engine = AnimationEngine(config, sim_config)
    
    try:
        print("🎯 Running fallback test...")
        # Set the simulator in the engine
        engine.simulator = simulator
        
        # Run the simulation
        results = engine.run_simulation()
        print("✅ Fallback test completed successfully")
        return True
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False


def main():
    """Run emoji animation tests."""
    print("=" * 50)
    print("🎮 RPS Emoji Animation Test Suite")
    print("=" * 50)
    
    # Test 1: Emoji rendering
    success1 = test_emoji_animation()
    
    # Test 2: Fallback rendering
    success2 = test_emoji_fallback()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Emoji rendering: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Fallback rendering: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Emoji rendering is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()