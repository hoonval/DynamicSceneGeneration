# -*- coding: utf-8 -*-
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import prior
from ai2thor.controller import Controller
import ai2thor
from PIL import Image
import copy
import hydra


def get_top_down_frame(controller, house_id, show_visualization):
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
    if show_visualization:
        # PIL Image를 numpy array로 변환
        top_down_array = np.array(top_down_frame)
        
        # 저장 디렉토리 생성
        save_dir = "data/scene_imgs"
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일명 생성 (타임스탬프 사용)
        import time
        timestamp = int(time.time())
        filename = f"scene_{house_id}.png"
        filepath = os.path.join(save_dir, filename)
        
        # PIL Image로 저장
        scene_image = Image.fromarray(top_down_array)
        scene_image.save(filepath)
        print(f"Scene image가 저장되었습니다: {filepath}")
        
        # 선택적으로 cv2로 시각화도 표시
        top_down_bgr = cv2.cvtColor(top_down_array, cv2.COLOR_RGB2BGR)
        height, width = top_down_bgr.shape[:2]
        scale = 0.8
        new_width = int(width * scale)
        new_height = int(height * scale)
        top_down_resized = cv2.resize(top_down_bgr, (new_width, new_height))
        
        # cv2.imshow('Procthor Top-Down View', top_down_resized)
        # print("이미지를 보려면 아무 키나 누르세요...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(filepath, top_down_resized)
    
    return Image.fromarray(top_down_frame)

@hydra.main(config_name="config", config_path="../configs", version_base=None)
def main(cfg):
    scenes = []
    scene_ids = []
    print(cfg)
    dataset = prior.load_dataset(cfg.dataset.name)
    
    for house_id in range(cfg.dataset.num_houses):
        # house_id = random.randint(0, len(dataset["train"]) - 1)
        # house_id = 2
        scene_ids.append(house_id)
        scenes.append(dataset["train"][house_id])
    
        controller = Controller(scene=scenes[house_id])
        top_down_frame = get_top_down_frame(controller, house_id, show_visualization=True)
    
    





if __name__ == "__main__":
    main()
