from ultralytics import YOLO
from pathlib import Path

WEIGHTS = r"C:\Users\yunse\ToI_02\ToI_yolo_runs\yolov8n_seg_case1_finetune_2ep_finetune_5ep\weights\best.pt"
DATASET_YAML = r"D:\ToI_Yolo_dataset\dataset.yaml"  # 네 yaml 경로로 수정

model = YOLO(WEIGHTS)

# test split으로 평가 (dataset.yaml에 test: 가 있어야 함)
metrics = model.val(
    data=DATASET_YAML,
    split="test",
    imgsz=640,         # 학습 때 IMG_SIZE와 동일하게
    batch=8,           # 환경에 맞게
    device="cpu",      # GPU면 "0"
    workers=0,
    plots=True,        # PR curve, confusion 등 저장
)

print("✅ Test eval done")
print("mAP50-95 (box):", metrics.box.map)
print("mAP50 (box):", metrics.box.map50)
print("mAP50-95 (mask):", metrics.seg.map)
print("mAP50 (mask):", metrics.seg.map50)