"""
Simple demo of emoji-based RPS animation.

This script demonstrates the emoji rendering system working with the RPS animation.
"""

import sys
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


def main():
    """Run a simple emoji RPS demo."""
    print("🎮 Starting RPS Emoji Animation Demo...")
    
    # Create emoji renderer and generate RPS emojis
    emoji_renderer = EmojiRenderer(default_size=40)
    emoji_paths = emoji_renderer.render_rps_emojis(size=40)
    
    print(f"✅ Generated emojis: {list(emoji_paths.keys())}")
    
    # Create animation configuration
    config = AnimationConfig(
        screen_width=800,
        screen_height=600,
        fps=30,  # Lower FPS for demo
        headless=False,
        export_frames=False,
        show_fps=True,
        show_stats=True
    )
    
    # Create simulation config
    sim_config = SimulationConfig(
        screen_width=800,
        screen_height=600,
        total_entities=20,  # Fewer entities for demo
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
        # Determine entity type
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
        
        # Enable emoji rendering
        emoji_path = emoji_paths.get(entity_type.value)
        if emoji_path:
            entity.emoji_image_path = emoji_path
            entity.use_emoji = True
            # Set the emoji path mapping for type conversions
            entity.set_emoji_path_mapping(emoji_paths)
            print(f"✅ Set emoji for {entity_type.value}: {emoji_path}")
        
        entities.append(entity)
    
    # Set entities in simulator
    simulator.entities = entities
    
    # Create animation engine
    engine = AnimationEngine(config, sim_config)
    
    print("🚀 Starting 10-second demo...")
    print("🪨 Rock (gray) vs 📄 Paper (white) vs ✂️ Scissors (red)")
    print("Close the window to stop early")
    
    try:
        # Set the simulator in the engine
        engine.simulator = simulator
        
        # Debug: Check entities before starting
        print(f"📊 Total entities before start: {len(simulator.entities)}")
        
        # Check distribution
        type_counts = {}
        for entity in simulator.entities:
            type_counts[entity.entity_type.value] = type_counts.get(entity.entity_type.value, 0) + 1
        
        print(f"📊 Entity distribution: {type_counts}")
        
        for i, entity in enumerate(simulator.entities[:3]):  # Show first 3
            print(f"  Entity {i}: {entity.entity_type.value}, emoji_path: {entity.emoji_image_path}, use_emoji: {entity.use_emoji}")
        
        # Check if simulation will end immediately
        if len(type_counts) == 1:
            print("⚠️ WARNING: All entities are the same type - simulation will end immediately!")
        
        # Run for a short time by modifying the max simulation time
        original_max_time = engine.simulation_config.max_simulation_time
        engine.simulation_config.max_simulation_time = 10.0  # 10 seconds
        
        # Add a temporary debug hook to start_simulation
        original_start = engine.simulator.start_simulation
        def debug_start_simulation():
            print(f"📊 Entities at start of start_simulation: {len(engine.simulator.entities)}")
            result = original_start()
            print(f"📊 Entities after start_simulation: {len(engine.simulator.entities)}")
            print(f"📊 Simulator is_running: {engine.simulator.is_running}")
            if engine.simulator.entities:
                print(f"  First entity: {engine.simulator.entities[0].entity_type.value}, emoji: {engine.simulator.entities[0].use_emoji}")
            return result
        
        engine.simulator.start_simulation = debug_start_simulation
        
        # Add debug hook to the main loop
        original_run = engine.run_simulation
        def debug_run_simulation():
            print(f"📊 Starting run_simulation...")
            print(f"📊 Engine running: {engine.running}")
            print(f"📊 Simulator is_running: {engine.simulator.is_running}")
            result = original_run()
            print(f"📊 Finished run_simulation with result: {result}")
            return result
        
        engine.run_simulation = debug_run_simulation
        
        # Run the simulation
        results = engine.run_simulation()
        
        # Restore original max time
        engine.simulation_config.max_simulation_time = original_max_time
        
        print("\n🏆 Demo Results:")
        
        # Extract results from nested structure
        sim_stats = results.get('simulation_stats', {})
        anim_stats = results.get('animation_stats', {})
        
        print(f"Duration: {sim_stats.get('simulation_time', 0):.2f} seconds")
        print(f"Total battles: {sim_stats.get('battles_fought', 0)}")
        print(f"Winner: {results.get('winner', 'None')}")
        print(f"Final counts: {sim_stats.get('entity_counts', {})}")
        print(f"Frames rendered: {anim_stats.get('frames_rendered', 0)}")
        print(f"Average FPS: {anim_stats.get('average_fps', 0):.1f}")
        
        # Show conversions
        conversions = sim_stats.get('conversions', {})
        if conversions:
            print(f"Conversions: {conversions}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
        return True
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Demo completed successfully!")
        print("The emoji rendering system is working correctly.")
    else:
        print("\n⚠️ Demo encountered issues.")