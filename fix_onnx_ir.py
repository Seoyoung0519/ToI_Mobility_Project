import onnx

SRC = r"C:\Users\yunse\ToI_02\ToI_yolo_runs\yolov8n_seg_case1_finetune_2ep_finetune_5ep\weights\best.onnx"
DST = r"C:\Users\yunse\ToI_02\ToI_yolo_runs\yolov8n_seg_case1_finetune_2ep_finetune_5ep\weights\best_ir11.onnx"

m = onnx.load(SRC)
print("before ir_version:", m.ir_version)

# ORT가 최대 11까지만 지원한다고 했으니 11로 맞춤
m.ir_version = 11
onnx.save(m, DST)

print("after ir_version:", onnx.load(DST).ir_version)
print("saved:", DST)
