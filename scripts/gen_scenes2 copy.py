# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
import hydra
from omegaconf import DictConfig
from pathlib import Path

import cv2
from PIL import Image
from models.ai2thor_scenes import AI2THORScenes
import prior
from ai2thor.controller import Controller


def get_top_down_frame(controller, scene_id, vis):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]

    # scene image를 data/scene_imgs 디렉토리에 저장
    if vis:
        # PIL Image를 numpy array로 변환
        top_down_array = np.array(top_down_frame)
        
        # 저장 디렉토리 생성 (절대 경로 사용)
        save_dir = Path("data/scene_imgs").resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성
        filename = f"scene_{scene_id}.png"
        filepath = save_dir / filename

        top_down_bgr = cv2.cvtColor(top_down_array, cv2.COLOR_RGB2BGR)
        height, width = top_down_bgr.shape[:2]
        scale = 0.8
        new_width = int(width * scale)
        new_height = int(height * scale)
        top_down_resized = cv2.resize(top_down_bgr, (new_width, new_height))
        
        # 이미지 저장 시도 및 디버깅
        cv2.imwrite(str(filepath), top_down_resized)
                
    return Image.fromarray(top_down_frame)

@hydra.main(config_name="config", config_path="../configs", version_base=None)
def main(cfg: DictConfig):

    dataset = prior.load_dataset(cfg.dataset.name)
    scene = dataset["train"][cfg.dataset.scene_id]
    
    # controller = Controller(scene=scene)
    # event = controller.step(action="GetSceneBounds")
    # objects = event.metadata.get("objects", [])
    # movable_objects = [obj for obj in objects if obj.get("pickupable", False)]
    # receptacle_objects = [obj for obj in objects if obj.get("receptacle", False)]

    ai2thor_scenes = AI2THORScenes(scene)

    SceneGenerator = SceneGenerator(ai2thor_scenes, cfg)


    top_down_frame = get_top_down_frame(controller, cfg.dataset.scene_id, vis=True)
    
    


if __name__ == "__main__":
    main()
