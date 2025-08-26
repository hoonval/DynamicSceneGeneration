# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
import hydra
from omegaconf import DictConfig
from pathlib import Path

import cv2
from PIL import Image

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
    
    controller = Controller(scene=scene)
    
    # 씬에 있는 모든 객체 조회 및 movable 객체 출력
    event = controller.step(action="GetSceneBounds")
    objects = event.metadata.get("objects", [])
    movable_objects = [obj for obj in objects if obj.get("pickupable", False)]

    # movable 객체들을 furniture 위에 배치
    if movable_objects:
        # 씬에서 receptacle 역할을 할 수 있는 furniture 찾기 (더 엄격한 검증)
        # Bed, Chair는 대부분 receptacle이 아니므로 제외
        valid_receptacle_types = ["Table", "CounterTop", "Shelf", "Desk", "Cabinet", "Drawer", "BookShelf", "NightStand", "CoffeeTable"]
        receptacles = [
            obj for obj in objects 
            if (obj.get("receptacle", False) and 
                obj.get("objectType") in valid_receptacle_types and
                not any(bad_name in obj.get("name", "").lower() for bad_name in ["bed", "chair", "sofa", "couch", "armchair"]))
        ]

        print(receptacles[0].get("name", "N/A"))
    
        if cfg.debug:
            print(f"\n사용 가능한 receptacle 객체들:")
            for i, rec in enumerate(receptacles):
                print(f"{i+1}. {rec.get('name', 'N/A')} - {rec.get('objectType', 'N/A')}")
        
        
        for i, movable_obj in enumerate(movable_objects):
            if i < len(receptacles):  # receptacle이 있는 경우에만
                receptacle = receptacles[i % len(receptacles)]  # receptacle 순환 사용
                
                if cfg.debug:
                    print(f"{movable_obj.get('name', 'N/A')}와 상호작용 가능한 포즈 찾는 중...")

                interactable_poses = controller.step(
                    action="GetInteractablePoses",
                    objectId=movable_obj["objectId"],
                    positions=None,
                    rotations=list(range(0, 360, 10)),
                    horizons=list(np.linspace(-30, 60, 30).astype(float)),
                    standings=[True, False] 
                ).metadata["actionReturn"]

                if interactable_poses:  # ✅ 올바른 변수명 사용
                    # 랜덤하게 하나의 포즈 선택
                    import random
                    pose = random.choice(interactable_poses)  # ✅ 올바른 변수명 사용
                    print('pose',pose)
                    
                    # TeleportFull로 선택된 포즈로 이동
                    controller.step("TeleportFull", **pose)
                    
                    # GetObjectsInFrame으로 현재 시야에 movable 객체가 보이는지 확인
                    if cfg.debug:
                        print(f"이동 후 {movable_obj.get('name', 'N/A')}가 시야에 보이는지 확인 중...")
                    
                    # pickup 전에 손에 있는 객체 확인 및 처리
                    agent_metadata = controller.last_event.metadata.get("agent", {})
                    if agent_metadata.get("heldObject"):
                        if cfg.debug:
                            print(f"⚠ 손에 이미 {agent_metadata['heldObject']}가 있습니다. 드롭합니다.")
                        # 현재 손에 있는 객체를 드롭
                        # controller.step(action="DropHandObject")
                    
                    # Pickup 시도
                    if cfg.debug:
                        print(f"Pickup 시도: {movable_obj.get('name', 'N/A')}")
                    
                    pickup_event = controller.step(
                        action="PickupObject",
                        objectId=movable_obj["objectId"],
                        forceAction=True,
                        manualInteract=False
                    )
                    
                    if pickup_event.metadata["lastActionSuccess"]:
                        if cfg.debug:
                            print(f"{receptacle.get('name', 'N/A')}와 상호작용 가능한 포즈 찾는 중...")
                        
                        receptacle_interactable_poses = controller.step(
                            action="GetInteractablePoses",
                            objectId=receptacle["objectId"],
                            positions=None,
                            rotations=list(range(0, 360, 45)),
                            horizons=list(np.linspace(-30, 60, 10).astype(float)),
                            standings=[True, False]
                        ).metadata["actionReturn"]
                        # print("receptacle :",receptacle["objectId"])

                        if receptacle_interactable_poses:
                            # receptacle과 상호작용 가능한 포즈로 이동
                            receptacle_pose = random.choice(receptacle_interactable_poses)

                            # TeleportFull로 receptacle 포즈로 이동
                            controller.step("TeleportFull", **receptacle_pose)
                            
                            # receptacle이 프레임에 보이는지 확인
                            receptacle_frame_event = controller.step(
                                action="GetObjectInFrame",
                                x=receptacle_pose["x"],
                                y=receptacle_pose["y"],
                                checkVisible=False,
                            )
                            receptacle_frame_objects = receptacle_frame_event.metadata.get("objects", [])
                            
                            receptacle_visible = False
                            for rec_obj in receptacle_frame_objects:
                                if rec_obj["objectId"] == receptacle["objectId"]:
                                    receptacle_visible = True
                                    break
                            
                            if receptacle_visible:
                                if cfg.debug:
                                    print(f"✓ {receptacle.get('name', 'N/A')}가 프레임에 보입니다.")
                                
                                # PutObject 시도 (receptacle 위에 배치)
                                receptacle_pos = receptacle.get("position", {})
                                put_event = controller.step(
                                    action="PutObject",
                                    x=receptacle_pos.get("x", 0.5),
                                    y=receptacle_pos.get("y", 0.5),
                                    putNearXY=True,  # 지정된 좌표 근처에 배치
                                    forceAction=True,
                                    placeStationary=True
                                )
                                
                                if cfg.debug:
                                    if put_event.metadata["lastActionSuccess"]:
                                        print(f"✓ {movable_obj.get('name', 'N/A')}를 {receptacle.get('name', 'N/A')} 위에 배치 성공")
                                    else:
                                        error_msg = put_event.metadata.get('errorMessage', '')
                                        print(f"✗ {movable_obj.get('name', 'N/A')}를 {receptacle.get('name', 'N/A')} 위에 배치 실패")
                                        print(f"    오류: {error_msg}")
                                        print(f"    Event metadata: {put_event.metadata}")
                                        print(f"    Receptacle: {receptacle.get('name')} ({receptacle.get('objectType')})")
                                        print(f"    Movable object: {movable_obj.get('name')} ({movable_obj.get('objectType')})")
                            else:
                                if cfg.debug:
                                    print(f"⚠ {receptacle.get('name', 'N/A')}가 프레임에 보이지 않습니다.")
                        else:
                            if cfg.debug:
                                print(f"⚠ {receptacle.get('name', 'N/A')}와 상호작용 가능한 포즈를 찾을 수 없습니다.")
                    else:
                        if cfg.debug:
                            error_msg = pickup_event.metadata.get('errorMessage', '')
                            print(f"✗ Pickup 실패: {movable_obj.get('name', 'N/A')}")
                            print(f"    오류: {error_msg}")
                else:
                    if cfg.debug:
                        print(f"⚠ {movable_obj.get('name', 'N/A')}와 상호작용 가능한 포즈를 찾을 수 없습니다.")
                    continue
                    

    top_down_frame = get_top_down_frame(controller, cfg.dataset.scene_id, vis=True)
    
    


if __name__ == "__main__":
    main()
