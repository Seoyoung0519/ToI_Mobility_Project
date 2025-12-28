# onnx_seg_score_img.py
import os
import cv2
import numpy as np
import onnxruntime as ort

# =========================
# 1) Letterbox (YOLO 스타일)
# =========================
def letterbox(im, new_shape=1088, color=(114, 114, 114)):
    """
    Resize + pad to square(new_shape) keeping aspect ratio.
    Return:
      - img_lb: letterboxed image
      - ratio: (r, r)
      - (dw, dh): padding added (left/right, top/bottom total/2 concept)
    """
    h0, w0 = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, (r, r), (left, top)  # pad is (left, top)

def scale_coords_xyxy_from_letterbox(xyxy, orig_shape, ratio, pad):
    """
    xyxy: (K,4) in letterbox coordinates (imgsz x imgsz)
    -> map back to original image coordinates
    """
    left, top = pad
    r = ratio[0]
    xyxy = xyxy.copy()
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy[:, :4] /= r

    h0, w0 = orig_shape[:2]
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w0 - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h0 - 1)
    return xyxy

def unpad_and_resize_mask_to_original(mask_lb, orig_shape, ratio, pad):
    """
    mask_lb: (H_lb, W_lb) in letterboxed resolution (imgsz x imgsz) (binary or float)
    -> remove padding, then resize back to original (h0,w0)
    """
    h0, w0 = orig_shape[:2]
    left, top = pad
    r = ratio[0]

    # unpad area size (the resized image region inside the letterbox)
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    # crop padding out of letterboxed mask
    cropped = mask_lb[top:top + new_h, left:left + new_w]

    # resize to original
    mask_orig = cv2.resize(cropped.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)
    return mask_orig

# =========================
# 2) NMS (numpy)
# =========================
def box_iou_xyxy(a, b):
    """
    a: (K,4), b:(M,4) -> IoU (K,M)
    """
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_xyxy(boxes, scores, iou_thres=0.5):
    """
    boxes:(K,4), scores:(K,)
    return keep indices
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou_xyxy(boxes[i:i+1], boxes[idxs[1:]]).reshape(-1)
        idxs = idxs[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int64)

# =========================
# 3) YOLOv8-seg ONNX 디코딩
# =========================
def sigmoid(x):
    x = np.clip(x, -50, 50)  # overflow 방지
    return 1.0 / (1.0 + np.exp(-x))

def ensure_pred_layout(pred):
    """
    Normalize pred tensor to shape (N, C) where C = 4 + 1 + nc + nm
    Common outputs:
      - (1, C, N)
      - (1, N, C)
      - (C, N)
      - (N, C)
    """
    pred = np.asarray(pred)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 2:
        # could be (C,N) or (N,C)
        if pred.shape[0] < pred.shape[1]:  # heuristic: C(=~38) < N(=24276)
            pred = pred.T  # (N,C)
        return pred
    raise ValueError(f"Unexpected pred ndim/shape: {pred.shape}")

def decode_yolov8_seg_onnx(outputs, nc, nm=32, conf_thres=0.25, iou_thres=0.5):
    """
    outputs: list of np arrays from onnxruntime
    Return:
      det_boxes_xyxy (K,4) in letterbox scale
      det_cls (K,)
      det_scores (K,)
      det_mask_coeff (K,nm)
      proto (nm,mh,mw)
    """
    arrs = [np.asarray(o) for o in outputs]

    proto = None
    pred = None
    for a in arrs:
        if a.ndim == 4:  # proto is 4D
            proto = a
        else:
            pred = a

    if proto is None or pred is None:
        raise RuntimeError(f"Could not identify pred/proto from outputs shapes: {[a.shape for a in arrs]}")

    # proto: (1, nm, mh, mw) -> (nm, mh, mw)
    if proto.ndim == 4 and proto.shape[0] == 1:
        nm_proto = proto.shape[1]
        proto = proto[0]  # (nm, mh, mw)
    else:
        raise ValueError(f"Unexpected proto shape: {proto.shape}")

    # ✅ nm은 proto 기준으로 확정
    nm = nm_proto

    # pred: normalize to (N, C)
    pred = ensure_pred_layout(pred)  # (N,C)
    C = pred.shape[1]

    # 기대 컬럼: 4 + 1 + nc + nm
    expected_min = 4 + 1 + nc
    if C < expected_min:
        raise ValueError(f"pred C={C} too small for nc={nc}. pred shape={pred.shape}")

    # ✅ mask coeff는 '마지막 nm개'로 안전하게 자르기
    # (export 옵션/버전에 따라 중간 구성 바뀌어도 마지막은 mask coeff인 경우가 일반적)
    mask_coeff = pred[:, -nm:]  # (N, nm)

    # cls logits은 5 ~ 5+nc 구간으로 고정
    xywh = pred[:, 0:4]
    obj = sigmoid(pred[:, 4])
    cls_logits = pred[:, 5:5+nc]
    cls_probs = sigmoid(cls_logits)

    cls_id = np.argmax(cls_probs, axis=1)
    cls_score = cls_probs[np.arange(cls_probs.shape[0]), cls_id]
    scores = obj * cls_score

    # filter by conf
    keep = scores > conf_thres
    if not np.any(keep):
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.float32),
                np.zeros((0, nm), np.float32),
                proto)

    xywh = xywh[keep]
    scores_f = scores[keep]
    cls_id = cls_id[keep]
    mask_coeff = mask_coeff[keep]

    # xywh -> xyxy (letterbox space)
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    # class-aware NMS
    final_idx = []
    for c in np.unique(cls_id):
        idx_c = np.where(cls_id == c)[0]
        keep_c = nms_xyxy(boxes[idx_c], scores_f[idx_c], iou_thres=iou_thres)
        final_idx.append(idx_c[keep_c])
    final_idx = np.concatenate(final_idx, axis=0)
    final_idx = final_idx[np.argsort(scores_f[final_idx])[::-1]]

    return boxes[final_idx], cls_id[final_idx], scores_f[final_idx], mask_coeff[final_idx], proto

def build_instance_masks(mask_coeff, proto):
    """
    mask_coeff: (K,nm)
    proto: (nm,mh,mw)
    Return masks: (K,mh,mw) in proto resolution, float in [0,1]
    """
    nm, mh, mw = proto.shape
    K = mask_coeff.shape[0]
    proto_flat = proto.reshape(nm, -1)           # (nm, mh*mw)
    masks = mask_coeff @ proto_flat              # (K, mh*mw)
    masks = masks.reshape(K, mh, mw)
    masks = sigmoid(masks)
    return masks

def imread_unicode(path: str):
    # Windows 한글 경로 대응: 파일을 바이트로 읽어서 imdecode
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# =========================
# 4) end-to-end: score_img
# =========================
def score_img_from_onnx(
    onnx_path: str,
    image_path: str,
    imgsz: int = 1088,
    class_names=None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    mask_thres: float = 0.5,
):
    """
    Return:
      score_all: union(all classes) area ratio in original image [0,1]
      score_by_class: dict {class_id: ratio}
    """
    if class_names is None:
        class_names = ["class0", "class1"]  # placeholder

    # load image
    im0 = imread_unicode(image_path)
    if im0 is None:
        raise RuntimeError(f"OpenCV failed to decode image (maybe corrupt): {image_path}")
    h0, w0 = im0.shape[:2]

    # letterbox
    im_lb, ratio, pad = letterbox(im0, new_shape=imgsz)
    # BGR->RGB, HWC->CHW, normalize 0..1
    x = im_lb[:, :, ::-1].astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,imgsz,imgsz)

    # run onnx
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(out_names, {in_name: x})

    # debug shapes
    print("ONNX output shapes:", [o.shape for o in outputs])

    nc = len(class_names)
    # decode
    boxes_lb, cls_ids, scores, mask_coeff, proto = decode_yolov8_seg_onnx(
        outputs, nc=nc, nm=32, conf_thres=conf_thres, iou_thres=iou_thres
    )

    if boxes_lb.shape[0] == 0:
        return 0.0, {i: 0.0 for i in range(nc)}

    # instance masks in proto resolution (K, mh, mw)
    masks_proto = build_instance_masks(mask_coeff, proto)  # float [0,1], shape (K,272,272)

    # class-wise union in proto resolution
    mh, mw = masks_proto.shape[1], masks_proto.shape[2]
    score_by_class = {}
    union_all_proto = np.zeros((mh, mw), dtype=np.uint8)

    for c in range(nc):
        idx = np.where(cls_ids == c)[0]
        if idx.size == 0:
            score_by_class[c] = 0.0
            continue

        # ✅ proto 해상도에서 union (K개 OR)
        union_c_proto = np.zeros((mh, mw), dtype=np.uint8)
        for i in idx:
            union_c_proto |= (masks_proto[i] > mask_thres).astype(np.uint8)

        union_all_proto |= union_c_proto

        # ✅ 여기서 딱 1번만 1088로 업샘플
        union_c_lb = cv2.resize(union_c_proto.astype(np.float32), (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        union_c_lb = (union_c_lb > 0.5).astype(np.uint8)

        # 원본 해상도로 복원 후 면적
        union_c_orig = unpad_and_resize_mask_to_original(union_c_lb, im0.shape, ratio, pad)
        union_c_orig_bin = (union_c_orig > 0.5).astype(np.uint8)
        score_by_class[c] = int(union_c_orig_bin.sum()) / float(h0 * w0)

    # all classes union score
    union_all_lb = cv2.resize(union_all_proto.astype(np.float32), (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    union_all_lb = (union_all_lb > 0.5).astype(np.uint8)

    union_all_orig = unpad_and_resize_mask_to_original(union_all_lb, im0.shape, ratio, pad)
    union_all_orig_bin = (union_all_orig > 0.5).astype(np.uint8)
    score_all = int(union_all_orig_bin.sum()) / float(h0 * w0)

    return score_all, score_by_class

# =========================
# 5) Example run
# =========================
if __name__ == "__main__":
    ONNX_PATH = r"C:\Users\yunse\ToI_02\ToI_yolo_runs\yolov8n_seg_case1_finetune_2ep_finetune_5ep\weights\best_ir11.onnx"
    IMG_PATH  = r"D:\ToI_Yolo_dataset\images\val\지자체도로정비AI데이터_CASE1(도로)_SUNNY_01\AM_sunny_CI01_20211013_094154_34_2.jpg"

    # 네 클래스명에 맞게 바꿔줘
    CLASS_NAMES = ["도로균열", "도로(홀)"]

    score_all, score_by_class = score_img_from_onnx(
        ONNX_PATH, IMG_PATH,
        imgsz=1088,
        class_names=CLASS_NAMES,
        conf_thres=0.25,
        iou_thres=0.5,
        mask_thres=0.5,
    )

    print("score_all:", score_all)
    for cid, s in score_by_class.items():
        print(f"class {cid} ({CLASS_NAMES[cid]}): {s}")
