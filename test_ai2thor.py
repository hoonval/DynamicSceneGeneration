#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import prior
from models.ai2thor_scenes import AI2THORScenes
from omegaconf import DictConfig

def test_ai2thor_scenes():
    """Test AI2THORScenes functionality with simpler config"""
    
    print("=== Testing AI2THORScenes ===")
    
    # Create a simple config
    cfg = DictConfig({
        'debug': True,
        'dataset': {
            'name': 'procthor-10k',
            'scene_id': 0
        }
    })
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = prior.load_dataset(cfg.dataset.name)
        scene = dataset["train"][cfg.dataset.scene_id]
        print(f"✓ Dataset loaded successfully")
        
        # Initialize AI2THORScenes
        print("Initializing AI2THORScenes...")
        ai2thor_scenes = AI2THORScenes(scene, cfg)
        print(f"✓ AI2THORScenes initialized successfully")
        
        # Test object information
        print(f"✓ Total objects: {len(ai2thor_scenes.objects)}")
        print(f"✓ Movable objects: {len(ai2thor_scenes.movable_objects)}")
        print(f"✓ Receptacle objects: {len(ai2thor_scenes.receptacle_objects)}")
        
        # Test object pickup if movable objects exist
        if ai2thor_scenes.movable_objects:
            print("\n=== Testing Object Pickup ===")
            test_obj_idx = 0
            test_obj = ai2thor_scenes.movable_objects[test_obj_idx]
            print(f"Testing pickup for object: {test_obj.get('name', 'Unknown')}")
            
            try:
                # Find object index in main objects list
                obj_index = next(i for i, obj in enumerate(ai2thor_scenes.objects) if obj['objectId'] == test_obj['objectId'])
                ai2thor_scenes.pickup_object(obj_index)
                print(f"✓ Pickup method executed successfully")
            except Exception as e:
                print(f"⚠ Pickup failed: {e}")
        
        # Test interactable poses for first object
        if ai2thor_scenes.objects:
            print("\n=== Testing Interactable Poses ===")
            test_obj_idx = 0
            try:
                poses = ai2thor_scenes.get_interactable_poses(test_obj_idx)
                print(f"✓ Found {len(poses)} interactable poses")
            except Exception as e:
                print(f"⚠ Interactable poses test failed: {e}")
        
        print("\n=== All tests completed ===")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up controller if it exists
        if 'ai2thor_scenes' in locals() and hasattr(ai2thor_scenes, 'controller'):
            ai2thor_scenes.controller.stop()
            print("Controller stopped")

if __name__ == "__main__":
    test_ai2thor_scenes()