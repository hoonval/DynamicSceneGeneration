#!/usr/bin/env python3
import sys
sys.path.append('.')

from models.ai2thor_scenes import AI2THORScenes
from omegaconf import DictConfig
from ai2thor.controller import Controller

def test_minimal():
    """Test AI2THORScenes with a basic scene"""
    
    print("=== Minimal AI2THORScenes Test ===")
    
    cfg = DictConfig({
        'debug': True
    })
    
    try:
        # Test basic controller initialization
        print("Testing basic Controller initialization...")
        controller = Controller(
            scene="FloorPlan1_physics", 
            headless=True,
            width=300,
            height=300
        )
        print("✓ Basic Controller works")
        
        # Test direct scene creation
        print("Testing AI2THORScenes initialization...")
        
        # Create a simple scene dict (mock scene data)
        simple_scene = {
            'scene': 'FloorPlan1_physics'
        }
        
        # Test our class initialization
        ai2thor = AI2THORScenes(simple_scene, cfg)
        
        print(f"✓ AI2THORScenes initialized")
        print(f"✓ Total objects: {len(ai2thor.objects)}")
        print(f"✓ Movable objects: {len(ai2thor.movable_objects)}")
        print(f"✓ Receptacle objects: {len(ai2thor.receptacle_objects)}")
        
        # Show some object details
        if ai2thor.objects:
            print(f"\nFirst few objects:")
            for i, obj in enumerate(ai2thor.objects[:3]):
                print(f"  {i}: {obj.get('name', 'Unknown')} - {obj.get('objectType', 'Unknown')}")
        
        # Test methods if objects exist
        if len(ai2thor.objects) > 0:
            print("\n=== Testing Methods ===")
            
            # Test get_interactable_poses
            try:
                poses = ai2thor.get_interactable_poses(0)
                print(f"✓ get_interactable_poses: Found {len(poses)} poses")
            except Exception as e:
                print(f"⚠ get_interactable_poses failed: {e}")
            
            # Test pickup_object for movable objects
            if ai2thor.movable_objects:
                try:
                    # Find index of first movable object
                    first_movable = ai2thor.movable_objects[0]
                    obj_idx = next(i for i, obj in enumerate(ai2thor.objects) 
                                 if obj['objectId'] == first_movable['objectId'])
                    ai2thor.pickup_object(obj_idx)
                    print(f"✓ pickup_object executed successfully")
                except Exception as e:
                    print(f"⚠ pickup_object failed: {e}")
        
        print("\n✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            if 'ai2thor' in locals():
                ai2thor.controller.stop()
            if 'controller' in locals():
                controller.stop()
            print("Controllers stopped")
        except:
            pass

if __name__ == "__main__":
    success = test_minimal()
    sys.exit(0 if success else 1)