"""
Debug live rendering to see exactly what's happening during animation.

This script adds debugging to the render method to track emoji rendering in real-time.
"""

import sys
import pygame
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.animation.rps_simulator import RPSSimulator, SimulationConfig
from lib.animation.animation_engine import AnimationEngine, AnimationConfig
from lib.animation.entity import RPSEntity, EntityType
from tools.animation.emoji_renderer import EmojiRenderer
from lib.utils.logging_config import get_logger

logger = get_logger(__name__)


def debug_render_method():
    """Add debugging to the render method."""
    
    # Store original render method
    original_render = RPSEntity.render
    
    render_call_count = 0
    
    def debug_render(self, screen):
        nonlocal render_call_count
        render_call_count += 1
        
        # Debug every 30th render call to avoid spam
        if render_call_count % 30 == 0:
            print(f"🔍 Render call #{render_call_count}: {self.entity_type.value}")
            print(f"   use_emoji: {self.use_emoji}")
            print(f"   emoji_image_path: {self.emoji_image_path}")
            print(f"   _emoji_surface: {self._emoji_surface is not None}")
            
            if self.use_emoji:
                if self._emoji_surface:
                    print(f"   → Using emoji rendering")
                else:
                    print(f"   → Emoji surface missing - using fallback")
            else:
                print(f"   → Using circle rendering")
        
        # Call original render method
        return original_render(self, screen)
    
    # Replace render method
    RPSEntity.render = debug_render
    
    return lambda: setattr(RPSEntity, 'render', original_render)


def main():
    """Run live rendering debug."""
    print("🔍 Live Rendering Debug")
    print("=" * 50)
    
    # Generate emojis
    renderer = EmojiRenderer(default_size=40)
    emoji_paths = renderer.render_rps_emojis(size=40)
    
    print(f"✅ Generated emojis: {list(emoji_paths.keys())}")
    
    # Create animation configuration
    config = AnimationConfig(
        screen_width=800,
        screen_height=600,
        fps=30,
        headless=False,
        export_frames=False,
        show_fps=True,
        show_stats=True
    )
    
    # Create simulation config
    sim_config = SimulationConfig(
        screen_width=800,
        screen_height=600,
        total_entities=12,  # Smaller number for easier debugging
        entity_distribution={
            EntityType.ROCK: 0.33,
            EntityType.PAPER: 0.33,
            EntityType.SCISSORS: 0.34
        }
    )
    
    # Create entities with emoji rendering
    entities = []
    for i in range(sim_config.total_entities):
        # Simple distribution
        entity_type = [EntityType.ROCK, EntityType.PAPER, EntityType.SCISSORS][i % 3]
        
        entity = RPSEntity.create_random(
            screen_width=config.screen_width,
            screen_height=config.screen_height,
            entity_type=entity_type
        )
        
        # Set emoji
        emoji_path = emoji_paths.get(entity_type.value)
        if emoji_path:
            entity.emoji_image_path = emoji_path
            entity.use_emoji = True
            entity.set_emoji_path_mapping(emoji_paths)
            print(f"✅ Entity {i}: {entity_type.value} with emoji")
        
        entities.append(entity)
    
    # Create simulator and set entities
    simulator = RPSSimulator(sim_config)
    simulator.entities = entities
    
    # Add debug to render method
    restore_render = debug_render_method()
    
    # Create animation engine
    engine = AnimationEngine(config, sim_config)
    engine.simulator = simulator
    
    try:
        print("\n🚀 Starting debug animation...")
        print("Watch the console for render debugging info")
        print("Animation will run for 5 seconds")
        
        # Limit simulation time
        engine.simulation_config.max_simulation_time = 5.0
        
        # Run simulation
        results = engine.run_simulation()
        
        print("\n📊 Debug Results:")
        sim_stats = results.get('simulation_stats', {})
        print(f"Duration: {sim_stats.get('simulation_time', 0):.2f} seconds")
        print(f"Frames rendered: {results.get('animation_stats', {}).get('frames_rendered', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original render method
        restore_render()


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Debug completed")
    else:
        print("\n❌ Debug failed")