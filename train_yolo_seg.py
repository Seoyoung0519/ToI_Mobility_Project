import os
from pathlib import Path
from ultralytics import YOLO
import csv

# =========================
# 공통 설정
# =========================

# 🔧 데이터 루트 경로 (D: 드라이브)
ROOT_DIR = Path(r"D:\ToI_Yolo_dataset")          # 반드시 실제 경로와 일치하게!
DATASET_YAML = ROOT_DIR / "dataset.yaml"

# YOLO 프로젝트 / 러닝 이름
PROJECT_NAME = "ToI_yolo_runs"
RUN_NAME = "yolov8n_seg_case1_finetune_2ep"

# 학습 하이퍼파라미터
EPOCHS = 5
BATCH_SIZE = 2
IMG_SIZE = 1088  # 1088x1088 letterbox

# wandb 끄기 (학습 전에 설정)
os.environ["WANDB_DISABLED"] = "true"


# =========================
# 1) dataset.yaml 생성 (없으면)
# =========================

def ensure_dataset_yaml():
    """
    D:/ToI_Yolo_dataset/dataset.yaml 을 생성 (이미 있으면 그대로 사용)
    """
    if DATASET_YAML.exists():
        print(f"✅ dataset.yaml 이미 존재: {DATASET_YAML}")
        with open(DATASET_YAML, "r", encoding="utf-8") as f:
            print("----[dataset.yaml 내용]----")
            print(f.read())
            print("---------------------------")
        return

    yaml_text = f"""path: {ROOT_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

nc: 2
names: ["도로균열", "도로(홀)"]
"""
    DATASET_YAML.write_text(yaml_text, encoding="utf-8")
    print(f"✅ dataset.yaml 생성 완료: {DATASET_YAML}")
    print(yaml_text)


# =========================
# 2) 데이터 존재 여부 간단 체크
# =========================

def check_dataset_structure():
    img_train = ROOT_DIR / "images" / "train"
    img_val = ROOT_DIR / "images" / "val"
    img_test = ROOT_DIR / "images" / "test"

    lbl_train = ROOT_DIR / "labels" / "train"
    lbl_val = ROOT_DIR / "labels" / "val"
    lbl_test = ROOT_DIR / "labels" / "test"

    def count_images(p: Path):
        if not p.exists():
            return 0
        files = list(p.rglob("*.jpg")) + list(p.rglob("*.jpeg")) + list(p.rglob("*.png"))
        return len(files)

    print("📂 이미지/라벨 경로 확인")
    print(" -", img_train, "이미지 개수:", count_images(img_train))
    print(" -", img_val,   "이미지 개수:", count_images(img_val))
    print(" -", img_test,  "이미지 개수:", count_images(img_test))
    print(" -", lbl_train, "라벨 폴더 존재:", lbl_train.exists())
    print(" -", lbl_val,   "라벨 폴더 존재:", lbl_val.exists())
    print(" -", lbl_test,  "라벨 폴더 존재:", lbl_test.exists())


# =========================
# 3) 최초 학습 함수
# =========================

def train_first():
    """
    yolov8n-seg.pt 로부터 처음 학습 시작
    """
    ensure_dataset_yaml()
    check_dataset_structure()

    print("✅ YOLOv8n-seg 모델 로드 중...")
    model = YOLO("yolov8n-seg.pt")
    print("✅ YOLOv8n-seg 모델 로드 완료")

    print(f"\n--- dataset.yaml 경로: {DATASET_YAML} ---\n")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",       # GPU 있으면 "cuda" 또는 "0" 로 변경
        workers=0,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        cache=False,

        # 증강 (Gaussian noise / motion blur 제외)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
    )

    print("✅ 학습 완료 (valid는 epoch마다 자동 평가)")
    return results


# =========================
# 4) 이어 학습(resume) 함수
# =========================
import pandas as pd

def resume_training(add_epochs: int = 2):
    """
        - 기존 run_dir의 last.pt / results.csv를 읽어 현재 epoch 파악
        - resume=True로 이어학습 시도
        - 이미 완료된 run이라 resume가 막히면 자동으로:
            resume=False로 last.pt에서 finetune을 '새 run'으로 시작 (add_epochs만큼)
        """
    run_dir = Path(PROJECT_NAME) / RUN_NAME
    last_weights = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    print("📌 run_dir:", run_dir.resolve())
    print("📌 last.pt:", last_weights.resolve(), "exists:", last_weights.exists())
    print("📌 results.csv:", results_csv.resolve(), "exists:", results_csv.exists())

    if not last_weights.exists():
        raise FileNotFoundError("❌ last.pt가 없습니다. 먼저 train_first()로 1회 학습을 시작해야 합니다.")

    # results.csv가 없을 수도 있으니 방어적으로 처리
    current_epoch = None
    if results_csv.exists():
        df = pd.read_csv(results_csv)

        # Ultralytics results.csv는 컬럼명에 공백이 섞여 나오는 경우가 있어 유연하게 찾기
        epoch_col = None
        for c in df.columns:
            if str(c).strip() == "epoch":
                epoch_col = c
                break
        if epoch_col is None:
            # 그래도 없으면, "epoch" 들어간 컬럼을 하나 잡기
            candidates = [c for c in df.columns if "epoch" in str(c)]
            epoch_col = candidates[0] if candidates else None

        if epoch_col is not None and len(df) > 0:
            # 보통 epoch는 0부터 기록되므로 마지막 값 + 1 = 현재 완료 epoch 수
            current_epoch = int(df[epoch_col].iloc[-1]) + 1

    if current_epoch is None:
        print("⚠️ results.csv에서 epoch를 읽지 못했습니다. (finetune 시 add_epochs만큼만 진행)")
    else:
        print(f"✅ 현재까지 완료된 epoch 추정: {current_epoch}")

    model = YOLO(str(last_weights))

    # 1) 먼저 resume=True로 "진짜 이어학습" 시도 (총 epoch 목표값으로 넣어야 함)
    #    단, Ultralytics가 이미 완료된 run이면 여기서 AssertionError가 날 수 있음
    if current_epoch is not None:
        target_total_epochs = current_epoch + add_epochs
    else:
        # epoch 정보를 못 읽으면 resume 모드로 총 epoch를 계산할 수 없으니 바로 finetune으로 유도
        target_total_epochs = None

    try:
        if target_total_epochs is None:
            raise RuntimeError("No epoch info -> skip resume attempt")

        print(f"🚀 resume=True로 시도: total_epochs={target_total_epochs} (현재 {current_epoch} + 추가 {add_epochs})")

        results = model.train(
            data=str(DATASET_YAML),
            epochs=target_total_epochs,   # '추가로' 학습할 epoch 수
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device="cpu",         # GPU 있으면 "cuda" 또는 "0"
            workers=0,
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True,
            cache=False,
            resume=True,           # 핵심: 기존 러닝 이어붙이기

            # 증강(네 설정 유지)
            hsv_h = 0.015,
            hsv_s = 0.7,
            hsv_v = 0.4,
            flipud = 0.0,
            fliplr = 0.5,
            scale = 0.5,
            shear = 0.0,
            perspective = 0.0,
            mosaic = 0.0,
            mixup = 0.0,
        )

        print("✅ resume=True 이어학습 완료")
        return results

    except Exception as e:
        msg = str(e)
        print("⚠️ resume=True가 실패했습니다. 사유:", msg)

        # 2) 자동 fallback: 이미 완료된 런이면 finetune(새 run)으로 add_epochs만큼만 학습
        #    (Ultralytics 에러 문구에 'nothing to resume'가 포함되는 경우가 흔함)
        new_run_name = f"{RUN_NAME}_finetune_{add_epochs}ep"
        print(f"🔁 fallback: resume=False로 finetune 새 run 시작 -> name='{new_run_name}', epochs={add_epochs}")

        results = model.train(
            data=str(DATASET_YAML),
            epochs=add_epochs,  # finetune에서는 '추가 epoch 수' 그대로
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device="cpu",
            workers=0,
            project=PROJECT_NAME,
            name=new_run_name,  # 새 run으로 저장(덮어쓰기 방지)
            exist_ok=True,
            cache=False,
            resume=False,

            # 증강(네 설정 유지)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.0,
            fliplr=0.5,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            mosaic=0.0,
            mixup=0.0,
        )

        print("✅ finetune 이어학습 완료")
        return results


# =========================
# 메인 실행부
# =========================

if __name__ == "__main__":
    # 1) 처음 학습할 때
    # train_first()

    # 2) 나중에 epoch 더 돌리고 싶으면, 위는 주석 처리하고 아래만 따로 실행
    resume_training(add_epochs=5)