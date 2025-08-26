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

    # pickup 가능한 object들을 receptacle object에 배치하는 함수
    def place_objects_on_receptacles(controller, movable_objects, receptacles):
        """
        pickup 가능한 object들을 receptacle object에 배치
        
        Args:
            controller: AI2-THOR controller
            movable_objects: pickup 가능한 object 리스트
            receptacles: receptacle object 리스트
        """
        if not movable_objects:
            if cfg.debug:
                print("⚠ pickup 가능한 object가 없습니다.")
            return
        
        if not receptacles:
            if cfg.debug:
                print("⚠ receptacle object가 없습니다.")
            return
        
        if cfg.debug:
            print(f"\n=== Object 배치 시작 ===")
            print(f"Pickup 가능한 objects: {len(movable_objects)}개")
            print(f"Receptacle objects: {len(receptacles)}개")
        
        # 각 pickup object를 receptacle에 배치
        for i, movable_obj in enumerate(movable_objects):
            # receptacle 순환 사용 (receptacle이 부족한 경우)
            receptacle = receptacles[i % len(receptacles)]
            
            if cfg.debug:
                print(f"\n--- {i+1}/{len(movable_objects)}: {movable_obj.get('name', 'N/A')} 처리 중 ---")
                print(f"대상 receptacle: {receptacle.get('name', 'N/A')} ({receptacle.get('objectType', 'N/A')})")
            
            # 1단계: movable object와 상호작용 가능한 포즈 찾기
            if cfg.debug:
                print(f"1단계: {movable_obj.get('name', 'N/A')}와 상호작용 가능한 포즈 찾는 중...")
            
            interactable_poses = controller.step(
                action="GetInteractablePoses",
                objectId=movable_obj["objectId"],
                positions=None,
                rotations=list(range(0, 360, 10)),
                horizons=list(np.linspace(-30, 60, 30).astype(float)),
                standings=[True, False] 
            ).metadata["actionReturn"]
            
            if not interactable_poses:
                if cfg.debug:
                    print(f"⚠ {movable_obj.get('name', 'N/A')}와 상호작용 가능한 포즈를 찾을 수 없습니다.")
                continue
            
            # 2단계: 상호작용 가능한 포즈로 이동
            import random
            pose = random.choice(interactable_poses)
            
            if cfg.debug:
                print(f"2단계: 선택된 포즈로 이동 - x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}")
            
            controller.step("TeleportFull", **pose)
            
            # 3단계: 손에 있는 객체 확인 및 처리
            agent_metadata = controller.last_event.metadata.get("agent", {})
            if agent_metadata.get("heldObject"):
                if cfg.debug:
                    print(f"3단계: 손에 이미 {agent_metadata['heldObject']}가 있습니다. 드롭합니다.")
                controller.step(action="DropHandObject")
            
            # 4단계: Pickup 시도
            if cfg.debug:
                print(f"4단계: {movable_obj.get('name', 'N/A')} Pickup 시도...")
            
            pickup_event = controller.step(
                action="PickupObject",
                objectId=movable_obj["objectId"],
                forceAction=True,
                manualInteract=False
            )
            
            if not pickup_event.metadata["lastActionSuccess"]:
                error_msg = pickup_event.metadata.get('errorMessage', '')
                if cfg.debug:
                    print(f"✗ Pickup 실패: {error_msg}")
                continue
            
            if cfg.debug:
                print(f"✓ Pickup 성공: {movable_obj.get('name', 'N/A')}")
            
            # 5단계: receptacle과 상호작용 가능한 포즈 찾기
            if cfg.debug:
                print(f"5단계: {receptacle.get('name', 'N/A')}와 상호작용 가능한 포즈 찾는 중...")
            
            receptacle_interactable_poses = controller.step(
                action="GetInteractablePoses",
                objectId=receptacle["objectId"],
                positions=None,
                rotations=list(range(0, 360, 45)),
                horizons=list(np.linspace(-30, 60, 10).astype(float)),
                standings=[True, False]
            ).metadata["actionReturn"]
            
            if not receptacle_interactable_poses:
                if cfg.debug:
                    print(f"⚠ {receptacle.get('name', 'N/A')}와 상호작용 가능한 포즈를 찾을 수 없습니다.")
                # pickup한 객체를 드롭
                controller.step(action="DropHandObject")
                continue
            
            # 6단계: receptacle 포즈로 이동
            receptacle_pose = random.choice(receptacle_interactable_poses)
            
            if cfg.debug:
                print(f"6단계: receptacle 포즈로 이동 - x={receptacle_pose['x']:.2f}, y={receptacle_pose['y']:.2f}, z={receptacle_pose['z']:.2f}")
            
            controller.step("TeleportFull", **receptacle_pose)
            
            # 7단계: receptacle이 시야에 보이는지 확인
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
            
            if not receptacle_visible:
                if cfg.debug:
                    print(f"⚠ {receptacle.get('name', 'N/A')}가 시야에 보이지 않습니다.")
                # pickup한 객체를 드롭
                controller.step(action="DropHandObject")
                continue
            
            if cfg.debug:
                print(f"✓ {receptacle.get('name', 'N/A')}가 시야에 보입니다.")
            
            # 8단계: PutObject 시도 (receptacle 위에 배치)
            if cfg.debug:
                print(f"8단계: {movable_obj.get('name', 'N/A')}를 {receptacle.get('name', 'N/A')} 위에 배치 시도...")
            
            receptacle_pos = receptacle.get("position", {})
            put_event = controller.step(
                action="PutObject",
                x=receptacle_pos.get("x", 0.5),
                y=receptacle_pos.get("y", 0.5),
                putNearXY=True,  # 지정된 좌표 근처에 배치
                forceAction=True,
                placeStationary=True
            )
            
            # 9단계: 결과 확인
            if put_event.metadata["lastActionSuccess"]:
                if cfg.debug:
                    print(f"🎉 성공: {movable_obj.get('name', 'N/A')}를 {receptacle.get('name', 'N/A')} 위에 배치 완료!")
            else:
                error_msg = put_event.metadata.get('errorMessage', '')
                if cfg.debug:
                    print(f"✗ 실패: {movable_obj.get('name', 'N/A')}를 {receptacle.get('name', 'N/A')} 위에 배치 실패")
                    print(f"    오류: {error_msg}")
                    print(f"    Event metadata: {put_event.metadata}")
                    print(f"    Receptacle: {receptacle.get('name')} ({receptacle.get('objectType')})")
                    print(f"    Movable object: {movable_obj.get('name')} ({movable_obj.get('objectType')})")
                # pickup한 객체를 드롭
                controller.step(action="DropHandObject")
        
        if cfg.debug:
            print(f"\n=== Object 배치 완료 ===")
    
    # 씬에서 receptacle 역할을 할 수 있는 furniture 찾기 (더 엄격한 검증)
    # Bed, Chair는 대부분 receptacle이 아니므로 제외
    valid_receptacle_types = ["Table", "CounterTop", "Shelf", "Desk", "Cabinet", "Drawer", "BookShelf", "NightStand", "CoffeeTable"]
    receptacles = [
        obj for obj in objects 
        if (obj.get("receptacle", False) and 
            obj.get("objectType") in valid_receptacle_types and
            not any(bad_name in obj.get("name", "").lower() for bad_name in ["bed", "chair", "sofa", "couch", "armchair"]))
    ]

    # if cfg.debug:
    #     print(f"\n사용 가능한 receptacle 객체들:")
    #     for i, rec in enumerate(receptacles):
    #         print(f"{i+1}. {rec.get('name', 'N/A')} - {rec.get('objectType', 'N/A')}")

    # pickup 가능한 object들을 receptacle에 배치
    place_objects_on_receptacles(controller, movable_objects, receptacles)                    

    top_down_frame = get_top_down_frame(controller, cfg.dataset.scene_id, vis=True)
    
    


if __name__ == "__main__":
    main()
