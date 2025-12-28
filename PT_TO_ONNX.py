from ultralytics import YOLO
from pathlib import Path

BEST_PT = r"C:\Users\yunse\ToI_02\ToI_yolo_runs\yolov8n_seg_case1_finetune_2ep_finetune_5ep\weights\best.pt"

def export_best_to_onnx(best_pt: str, imgsz: int = 1088, opset: int = 17, simplify: bool = True):
    model = YOLO(best_pt)

    # YOLOv8 export: 결과 onnx 경로를 반환함
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,        # 학습/추론 imgsz와 동일하게
        opset=opset,
        simplify=simplify,
        dynamic=False       # 고정 입력(권장). 필요하면 True로 바꿀 수 있음
    )

    print("✅ Exported ONNX:", onnx_path)
    return onnx_path

if __name__ == "__main__":
    export_best_to_onnx(BEST_PT, imgsz=1088)
