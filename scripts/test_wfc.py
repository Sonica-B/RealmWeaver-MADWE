"""
Test script for Day 6 Hierarchical WFC
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from wfc.hierarchical_wfc import HierarchicalWFC


def visualize_hierarchical_generation(world_data: dict):
    """Visualize the hierarchical generation results"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Region map
    ax = axes[0, 0]
    region_map = world_data['region_map']
    im1 = ax.imshow(region_map, cmap='tab10', interpolation='nearest')
    ax.set_title('Level 1: Regions')
    ax.set_xlabel('Forest=101, Desert=102, Cyberpunk=103')
    
    # 2. Combined biome map with proper colors
    ax = axes[0, 1]
    biome_visual = np.zeros((world_data['world_size'][1], world_data['world_size'][0], 3))
    
    # Use colors if available, otherwise use default colormap
    if 'biome_colors' in world_data:
        biome_colors = list(world_data['biome_colors'].values())
        for (x, y), biome_data in world_data['biome_maps'].items():
            biome_map = biome_data['map']
            for i in range(32):
                for j in range(32):
                    biome_idx = biome_map[i, j]
                    color = np.array(biome_colors[biome_idx]) / 255.0
                    biome_visual[y+i, x+j] = color
        ax.imshow(biome_visual, interpolation='nearest')
        ax.set_title('Level 2: Biomes (Green=Forest, Tan=Desert, Purple=Cyberpunk)')
    else:
        # Fallback for old version
        for (x, y), biome_data in world_data['biome_maps'].items():
            biome_map = biome_data['map']
            biome_visual[y:y+32, x:x+32, 0] = biome_map / 3.0
        ax.imshow(biome_visual, interpolation='nearest')
        ax.set_title('Level 2: Biomes')
    
    # 3. Sample tile map with colors
    ax = axes[1, 0]
    first_chunk_pos = list(world_data['tile_maps'].keys())[0]
    sample_tiles = world_data['tile_maps'][first_chunk_pos]
    
    if 'tile_colors' in world_data:
        # Create colored tile visualization
        tile_visual = np.zeros((32, 32, 3))
        tile_colors = world_data['tile_colors']
        
        for i in range(32):
            for j in range(32):
                tile_id = sample_tiles[i, j]
                color = np.array(tile_colors.get(tile_id, (128, 128, 128))) / 255.0
                tile_visual[i, j] = color
                
        ax.imshow(tile_visual, interpolation='nearest')
    else:
        # Fallback
        ax.imshow(sample_tiles, cmap='terrain', interpolation='nearest')
        
    ax.set_title(f'Level 3: Tiles at {first_chunk_pos}')
    
    # 4. Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""Hierarchical Generation Statistics:
    
World Size: {world_data['world_size']}
Hierarchy Levels: {world_data['hierarchy_levels']}
Region Map Size: {region_map.shape}
Number of Chunks: {len(world_data['biome_maps'])}
Tiles per Chunk: 32x32

Unique Regions: {len(np.unique(region_map))}
Total Biome Maps: {len(world_data['biome_maps'])}
Total Tile Maps: {len(world_data['tile_maps'])}
"""
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('hierarchical_wfc_result.png', dpi=150)
    plt.show()


def main():
    print("=== Day 6: Hierarchical WFC Test ===\n")
    
    # Create hierarchical WFC
    hwfc = HierarchicalWFC()
    
    # Test 1: Small world generation
    print("Test 1: Generating 256x256 world...")
    world_data = hwfc.generate_hierarchical_world(
        world_size=(256, 256),
        use_neural_constraints=False
    )
    
    print(f"✓ Generated world with {len(world_data['tile_maps'])} chunks")
    
    # Test 2: With neural constraints (placeholder)
    print("\nTest 2: Testing with neural constraints...")
    world_data_neural = hwfc.generate_hierarchical_world(
        world_size=(128, 128),
        use_neural_constraints=True
    )
    
    print(f"✓ Neural constraint generation complete")
    
    # Visualize results
    print("\nVisualizing hierarchical generation...")
    visualize_hierarchical_generation(world_data)
    
    # Test constraint propagation
    print("\nTesting constraint propagation:")
    
    # Check region → biome constraints
    for pos, biome_data in list(world_data['biome_maps'].items())[:3]:
        parent_region = biome_data['parent_region']
        dominant_biome = biome_data['dominant_biome']
        print(f"  Position {pos}: Region {parent_region} → Biome '{dominant_biome}'")


if __name__ == "__main__":
    main()