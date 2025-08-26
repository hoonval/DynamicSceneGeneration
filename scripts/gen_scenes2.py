# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import copy
import hydra
from omegaconf import DictConfig
from pathlib import Path

import cv2
from PIL import Image
from models import SceneController, SceneGenerator
from utils.vis import get_top_down_frame
from ai2thor.controller import Controller
import prior



@hydra.main(config_name="config", config_path="../configs", version_base=None)
def main(cfg: DictConfig):

    dataset = prior.load_dataset(cfg.dataset.name)
    scene = dataset["train"][cfg.dataset.scene_id]

    scene_controller = SceneController(cfg, scene)
    # scene_controller.show_scene_objects(0)
    # scene_controller.show_movable_objects()
    # scene_controller.show_receptacle_objects()
    scene_generator = SceneGenerator(scene_controller)

    # scene_controller.teleport_to_object(0)
    scene_generator.move_object('TissueBox|surface|6|53', 'DiningTable|4|1|0')
    top_down_frame = get_top_down_frame(scene_controller.controller, cfg.dataset.scene_id, vis=True)

if __name__ == "__main__":
    main()
