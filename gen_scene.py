import prior
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Optional
from ai2thor.controller import Controller

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcTHORSceneViewer:
    def __init__(self, width: int = 1600, height: int = 1200):
        self.width = width
        self.height = height
        self.controller = None
        
    def initialize_controller(self):
        """AI2Thor 컨트롤러 초기화"""
        if self.controller is None:
            logger.info("AI2Thor 컨트롤러 초기화 중...")
            self.controller = Controller(
                scene="Procedural",
                width=self.width,
                height=self.height,
                gridSize=1.0,
                snapToGrid=True,
                rotateStepDegrees=90,
                renderInstanceSegmentation=True,
                visibilityDistance=1.5
            )
            self.controller.step(action="Initialize")
            logger.info("컨트롤러 초기화 완료")
    
    def load_scene(self, split: str, index: int):
        """ProcTHOR 데이터셋에서 특정 씬을 로드"""
        try:
            dataset = prior.load_dataset("procthor-10k")
            house = dataset[split][index]
            
            logger.info(f"씬 로딩 중: {split}[{index}]")
            event = self.controller.step(action="CreateHouse", house=house)
            
            if not event.metadata["lastActionSuccess"]:
                raise RuntimeError(f"CreateHouse 실패: {event.metadata.get('errorMessage', 'Unknown error')}")
                
            logger.info(f"씬 {index} 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"씬 {index} 로딩 실패: {e}")
            return False
    
    def get_scene_bounds(self) -> Dict[str, float]:
        """씬의 바운딩 박스 계산"""
        objects = self.controller.last_event.metadata["objects"]
        positions = [obj["position"] for obj in objects if obj.get("position")]
        
        if not positions:
            return {"x_min": -5, "x_max": 5, "z_min": -5, "z_max": 5}
        
        x_coords = [pos["x"] for pos in positions]
        z_coords = [pos["z"] for pos in positions]
        
        return {
            "x_min": min(x_coords) - 2,
            "x_max": max(x_coords) + 2,
            "z_min": min(z_coords) - 2,
            "z_max": max(z_coords) + 2
        }
    
    def get_top_view_image(self, y_position: float = 4.0) -> np.ndarray:
        """씬의 top-view 이미지 생성"""
        # 현재 에이전트 상태 저장
        original_event = self.controller.last_event
        original_agent = original_event.metadata["agent"]
        
        # 씬 중앙 위치 계산
        bounds = self.get_scene_bounds()
        center_x = (bounds["x_min"] + bounds["x_max"]) / 2
        center_z = (bounds["z_min"] + bounds["z_max"]) / 2
        
        # top-down 시점으로 이동
        self.controller.step(
            action="TeleportFull",
            position={"x": center_x, "y": y_position, "z": center_z},
            rotation={"x": 90, "y": 0, "z": 0},
            horizon=90,
            standing=False
        )
        
        # 이미지 캡처
        event = self.controller.step(action="Pass")
        top_view_image = event.frame
        
        # 원래 위치로 복원
        self.controller.step(
            action="TeleportFull",
            position=original_agent["position"],
            rotation=original_agent["rotation"],
            horizon=original_agent["cameraHorizon"],
            standing=original_agent["isStanding"]
        )
        
        return top_view_image
    
    def close(self):
        """컨트롤러 종료"""
        if self.controller:
            self.controller.stop()
            self.controller = None

def select_random_scenes(split: str = "train", count: int = 1, max_index: int = 100, seed: int = 42) -> List[int]:
    """ProcTHOR 데이터셋에서 랜덤하게 씬 인덱스들을 선정"""
    random.seed(seed)
    
    if count > max_index:
        count = max_index
    
    available_indices = list(range(max_index))
    random.shuffle(available_indices)
    selected = available_indices[:count]
    
    logger.info(f"선정된 씬 인덱스들: {selected}")
    return selected

def show_scenes_top_view(
    scene_indices: List[int], 
    split: str = "train",
    save_path: Optional[str] = None,
    width: int = 1600,
    height: int = 1200
):
    """선정된 씬들의 top-view를 보여주는 함수"""
    
    viewer = ProcTHORSceneViewer(width, height)
    num_scenes = len(scene_indices)
    
    # subplot 그리드 계산
    cols = min(3, num_scenes)  # 최대 3열
    rows = (num_scenes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    
    # axes를 1차원 배열로 변환
    if num_scenes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    try:
        viewer.initialize_controller()
        
        for idx, scene_index in enumerate(scene_indices):
            ax = axes[idx]
            
            try:
                # 씬 로드
                success = viewer.load_scene(split, scene_index)
                
                if success:
                    # top-view 이미지 생성
                    top_view_img = viewer.get_top_view_image()
                    
                    ax.imshow(top_view_img)
                    ax.set_title(f"Scene {scene_index} ({split})", fontsize=12)
                    ax.axis('off')
                    
                    logger.info(f"씬 {scene_index} 렌더링 완료")
                else:
                    # 로딩 실패시 에러 표시
                    ax.text(0.5, 0.5, f"Failed to load\nScene {scene_index}", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
                    ax.set_title(f"Scene {scene_index} (Error)", fontsize=12)
                    ax.axis('off')
                    
            except Exception as e:
                logger.error(f"씬 {scene_index} 처리 중 오류: {e}")
                ax.text(0.5, 0.5, f"Error processing\nScene {scene_index}\n{str(e)[:50]}", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
                ax.set_title(f"Scene {scene_index} (Error)", fontsize=12)
                ax.axis('off')
        
        # 빈 subplot들 숨기기
        for idx in range(num_scenes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 이미지 저장
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"이미지 저장됨: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"씬 뷰어 실행 중 오류: {e}")
        raise
    finally:
        viewer.close()

def generate_random_scenes_topview(
    count: int = 5,
    split: str = "train", 
    max_index: int = 100,
    save_path: Optional[str] = None,
    seed: int = 42,
    width: int = 1600,
    height: int = 1200
) -> List[int]:
    """
    ProcTHOR 데이터셋에서 랜덤하게 씬을 선정하고 top-view로 보여주는 메인 함수
    
    Args:
        count: 선정할 씬의 개수
        split: 데이터셋 분할 ("train", "val", "test")
        max_index: 사용할 최대 씬 인덱스
        save_path: 결과 이미지 저장 경로 (None이면 저장하지 않음)
        seed: 랜덤 시드
        width: 렌더링 이미지 너비
        height: 렌더링 이미지 높이
        
    Returns:
        선정된 씬 인덱스들의 리스트
    """
    
    logger.info(f"ProcTHOR {split} 데이터셋에서 {count}개 씬을 랜덤 선정합니다...")
    logger.info(f"설정: max_index={max_index}, seed={seed}, 해상도={width}x{height}")
    
    # 랜덤 씬 선정
    selected_indices = select_random_scenes(split, count, max_index, seed)
    
    # top-view 렌더링 및 표시
    show_scenes_top_view(selected_indices, split, save_path, width, height)
    
    return selected_indices

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ProcTHOR 데이터셋에서 랜덤 씬 선정 및 Top-View 렌더링")
    parser.add_argument("--count", type=int, default=4, help="선정할 씬의 개수 (기본값: 4)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], 
                       help="데이터셋 split (기본값: train)")
    parser.add_argument("--max-index", type=int, default=100, 
                       help="사용할 최대 씬 인덱스 (기본값: 100)")
    parser.add_argument("--save-path", type=str, default=None, 
                       help="결과 이미지 저장 경로 (기본값: 저장하지 않음)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본값: 42)")
    parser.add_argument("--width", type=int, default=1600, help="렌더링 이미지 너비 (기본값: 1600)")
    parser.add_argument("--height", type=int, default=1200, help="렌더링 이미지 높이 (기본값: 1200)")
    
    args = parser.parse_args()
    
    try:
        # 랜덤 씬 생성 및 top-view 표시
        selected_scenes = generate_random_scenes_topview(
            count=args.count,
            split=args.split,
            max_index=args.max_index,
            save_path=args.save_path,
            seed=args.seed,
            width=args.width,
            height=args.height
        )
        
        print(f"\n✅ 완료! 선정된 씬들: {selected_scenes}")
        if args.save_path:
            print(f"📁 저장된 파일: {args.save_path}")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        print(f"\n❌ 오류: {e}")
        
    print("\n사용 예시:")
    print("python gen_scene.py --count 6 --split train --save-path scenes.png")
    print("python gen_scene.py --count 3 --split val --seed 123")