import os
import prior
import random
import numpy as np
from ai2thor.controller import Controller


class SceneController:
    def __init__(self, cfg, scene):
        self.cfg = cfg
        self.controller = Controller(scene=scene)
        event = self.controller.step(action="GetSceneBounds")
        self.objects = event.metadata.get("objects", [])
        self.movable_objects = [obj for obj in self.objects if obj.get("pickupable", False)]
        self.movable_list = {obj.get("objectId"): i for i, obj in enumerate(self.movable_objects)}
        self.receptacle_objects = [obj for obj in self.objects if obj.get("receptacle", False)]
        self.receptacle_list = {obj.get("objectId"): i for i, obj in enumerate(self.receptacle_objects)}


    def get_interactable_poses(self, object, is_movable=True):

        if is_movable:
            object_id = self.movable_objects[self.movable_list[object]]['objectId']
        else:
            object_id = self.receptacle_objects[self.receptacle_list[object]]['objectId']

        interactable_poses = self.controller.step(
            action="GetInteractablePoses",
            objectId=object_id,
            positions=None,
            rotations=list(range(0, 360, 10)),
            horizons=list(np.linspace(-30, 60, 30).astype(float)),
            standings=[True, False] 
        ).metadata["actionReturn"]

        return interactable_poses

    def pickup_object(self, object):
        object_id = self.movable_objects[self.movable_list[object]]['objectId']
        self.controller.step(action="PickupObject", objectId=object_id)

    def put_object(self, object):
        object_id = self.receptacle_objects[self.receptacle_list[object]]['objectId']
        self.controller.step(action="PutObject", objectId=object_id)

    def get_receptacle(self, object):
        object_id = self.receptacle_objects[self.receptacle_list[object]]['objectId']
        self.controller.step(action="GetReceptacle", objectId=object_id)

    def teleport_to_object(self, pose):
        pos = random.choice(pose)
        self.controller.step(action="TeleportFull", **pos)

    def __len__(self):
        return len(self.objects)

    def show_scene_objects(self):
        for i in range(len(self.objects)):
            print(self.objects[i]['name'])
 
    def show_movable_objects(self):
        for i in range(len(self.movable_objects)):
            print(self.movable_objects[i]['name'])

    def show_receptacle_objects(self):
        for i in range(len(self.receptacle_objects)):
            print(self.receptacle_objects[i]['name'])




