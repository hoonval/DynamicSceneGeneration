import random



class SceneGenerator:
    def __init__(self, scene_controller):
        self.scene_controller = scene_controller    

    def move_object(self, movable_object, receptacle_object):
        poses = self.scene_controller.get_interactable_poses(movable_object, is_movable=True)
        self.scene_controller.teleport_to_object(poses)
        self.scene_controller.pickup_object(movable_object)

        # receptacle로 이동하여 객체 배치
        receptacle_poses = self.scene_controller.get_interactable_poses(receptacle_object, is_movable=False)
        self.scene_controller.teleport_to_object(receptacle_poses)
        self.scene_controller.put_object(receptacle_object)
        