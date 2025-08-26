# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
import hydra
from omegaconf import DictConfig
from pathlib import Path

import cv2
from PIL import Image
from models import AI2THORScenes, SceneGenerator
from utils.vis import get_top_down_frame
from ai2thor.controller import Controller
import prior



@hydra.main(config_name="config", config_path="../configs", version_base=None)
def main(cfg: DictConfig):
    dataset = prior.load_dataset(cfg.dataset.name)
    scene = dataset["train"][cfg.dataset.scene_id]
    controller = Controller(scene=scene)
    # controller = Controller(scene=scene)
    # event = controller.step(action="GetSceneBounds")
    # objects = event.metadata.get("objects", [])
    # movable_objects = [obj for obj in objects if obj.get("pickupable", False)]
    # receptacle_objects = [obj for obj in objects if obj.get("receptacle", False)]
    ai2thor_scenes = AI2THORScenes(cfg, scene)
    print('dsadjaspjdpasjpdsjp')
    # Test the AI2THORScenes functionality
    print(f"Scene loaded with {len(ai2thor_scenes.objects)} total objects")
    print(f"Movable objects: {len(ai2thor_scenes.movable_objects)}")
    print(f"Receptacle objects: {len(ai2thor_scenes.receptacle_objects)}")
    
    # Test visualization
    top_down_frame = get_top_down_frame(ai2thor_scenes.controller, cfg.dataset.scene_id, vis=True)
    


if __name__ == "__main__":
    main()
