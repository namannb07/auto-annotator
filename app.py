#!/usr/bin/env python3
"""
Auto-Annotator — Streamlit Web App
YOLOv8 (Ultralytics/YOLOv8 on HuggingFace) • CPU inference
Features: upload images, auto-annotate, manual box edit, YOLO/COCO export
"""

import io
import json
import zipfile
import datetime
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto-Annotator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COCO-80 class names ───────────────────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

CLASS_PALETTE = [
    "#FF4444","#44DD44","#4488FF","#FFEE00","#FF44FF","#00DDDD",
    "#FF8800","#FF44AA","#AA44FF","#44FFAA","#FF6644","#44AAFF",
    "#FFAA44","#44FF66","#FF4488","#88FF44","#FF6600","#0066FF",
    "#FF0066","#66FF00",
]

MODEL_FILES = {
    "YOLOv8n (6 MB — fastest)":  "yolov8n.pt",
    "YOLOv8s (22 MB — balanced)": "yolov8s.pt",
    "YOLOv8m (52 MB — accurate)": "yolov8m.pt",
    "YOLOv8l (88 MB — best)":     "yolov8l.pt",
}

# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "model": None,
        "model_name": None,
        "images": {},           # filename → PIL Image
        "annotations": {},      # filename → List[dict]  {class_id, cx, cy, w, h, conf}
        "current_img": None,
        "conf_thresh": 0.25,
        "selected_box": None,
        "edit_mode": "view",    # view | add | delete
        "labels": COCO_CLASSES,
        "custom_classes": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    """Download from HuggingFace and load with Ultralytics."""
    from ultralytics import YOLO
    path = hf_hub_download(
        repo_id="Ultralytics/YOLOv8",
        filename=filename,
        cache_dir=tempfile.gettempdir(),
    )
    model = YOLO(path)
    model.to("cpu")
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(pil_img: Image.Image, conf: float) -> List[dict]:
    model = st.session_state.model
    if model is None:
        return []
    results = model.predict(
        source=np.array(pil_img.convert("RGB")),
        conf=conf,
        device="cpu",
        verbose=False,
    )
    boxes = []
    if results and results[0].boxes is not None:
        r = results[0]
        iw, ih = pil_img.size
        for box in r.boxes:
            xyxyn = box.xyxyn[0].tolist()
            x1n, y1n, x2n, y2n = xyxyn
            cx = (x1n + x2n) / 2
            cy = (y1n + y2n) / 2
            w  = x2n - x1n
            h  = y2n - y1n
            boxes.append({
                "class_id": int(box.cls[0]),
                "cx": cx, "cy": cy, "w": w, "h": h,
                "conf": float(box.conf[0]),
            })
    return boxes

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(pil_img: Image.Image, boxes: List[dict], selected_idx: Optional[int] = None) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    labels = st.session_state.labels

    for i, box in enumerate(boxes):
        cid = box["class_id"]
        color = CLASS_PALETTE[cid % len(CLASS_PALETTE)]
        x1 = int((box["cx"] - box["w"] / 2) * iw)
        y1 = int((box["cy"] - box["h"] / 2) * ih)
        x2 = int((box["cx"] + box["w"] / 2) * iw)
        y2 = int((box["cy"] + box["h"] / 2) * ih)

        lw = 4 if i == selected_idx else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = labels[cid] if cid < len(labels) else str(cid)
        conf_str = f"{box['conf']:.2f}" if box.get("conf", 1.0) < 1.0 else ""
        text = f"{label} {conf_str}".strip()

        # Label background
        tx, ty = x1, max(0, y1 - 18)
        tw = len(text) * 7 + 6
        draw.rectangle([tx, ty, tx + tw, ty + 18], fill=color)
        draw.text((tx + 3, ty + 2), text, fill="white")

        # Corner handles for selected
        if i == selected_idx:
            hs = 6
            for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                draw.rectangle([hx - hs, hy - hs, hx + hs, hy + hs], fill=color)

    return img

# ── Export helpers ────────────────────────────────────────────────────────────
def export_yolo_zip() -> bytes:
    buf = io.BytesIO()
    labels = st.session_state.labels
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # classes.txt
        zf.writestr("classes.txt", "\n".join(labels) + "\n")
        # data.yaml
        names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(labels))
        yaml_str = f"nc: {len(labels)}\nnames:\n{names_yaml}\n"
        zf.writestr("data.yaml", yaml_str)

        for fname, boxes in st.session_state.annotations.items():
            if not boxes:
                continue
            # image
            img_bytes = io.BytesIO()
            st.session_state.images[fname].save(img_bytes, format="JPEG")
            zf.writestr(f"images/{fname}", img_bytes.getvalue())
            # label
            lines = [f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes]
            stem = Path(fname).stem
            zf.writestr(f"labels/{stem}.txt", "\n".join(lines) + "\n")
    buf.seek(0)
    return buf.read()

def export_coco_zip() -> bytes:
    labels = st.session_state.labels
    categories = [{"id": i, "name": n, "supercategory": "object"} for i, n in enumerate(labels)]
    coco_images, coco_anns, ann_id = [], [], 0

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_id, (fname, boxes) in enumerate(st.session_state.annotations.items()):
            if not boxes:
                continue
            pil = st.session_state.images[fname]
            iw, ih = pil.size
            img_bytes = io.BytesIO()
            pil.save(img_bytes, format="JPEG")
            zf.writestr(f"images/{fname}", img_bytes.getvalue())
            coco_images.append({"id": img_id, "file_name": fname, "width": iw, "height": ih})
            for box in boxes:
                abs_w = box["w"] * iw
                abs_h = box["h"] * ih
                abs_x = box["cx"] * iw - abs_w / 2
                abs_y = box["cy"] * ih - abs_h / 2
                coco_anns.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": box["class_id"],
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "segmentation": [], "iscrowd": 0,
                })
                ann_id += 1

        payload = {
            "info": {"description": "Auto-Annotator Export", "version": "1.0",
                     "date_created": datetime.datetime.now().isoformat()},
            "images": coco_images,
            "annotations": coco_anns,
            "categories": categories,
        }
        zf.writestr("annotations.json", json.dumps(payload, indent=2))
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0f;
    color: #e8e8e8;
}

/* Header */
.header-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 4px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e94560;
    letter-spacing: -0.5px;
    margin: 0;
}
.header-sub {
    font-size: 0.8rem;
    color: #a0a0b0;
    font-family: 'Space Mono', monospace;
    margin: 4px 0 0 0;
}

/* Sidebar sections */
.sidebar-section {
    background: #1a1a2e;
    border-left: 3px solid #e94560;
    padding: 10px 14px;
    margin-bottom: 12px;
    border-radius: 0 4px 4px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #e94560;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Stats badge */
.stat-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin: 2px;
}

/* Box list item */
.box-item {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    cursor: pointer;
    transition: border-color 0.15s;
}
.box-item:hover { border-color: #e94560; }
.box-item.selected { border-color: #e94560; background: #2a1a2e; }

/* Instruction banner */
.instruction {
    background: #1a2a1a;
    border: 1px solid #44aa44;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.78rem;
    color: #88cc88;
    font-family: 'Space Mono', monospace;
    margin-bottom: 10px;
}

/* Override streamlit button */
.stButton > button {
    background: #e94560 !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    background: #c73350 !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stFileUploader"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #a0a0b0 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <div>
    <p class="header-title">🎯 AUTO-ANNOTATOR</p>
    <p class="header-sub">YOLOv8 · CPU inference · YOLO/COCO export</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">① Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select variant",
        list(MODEL_FILES.keys()),
        index=0,
    )
    if st.button("⬇ Load Model"):
        fname = MODEL_FILES[model_choice]
        with st.spinner(f"Downloading {fname} from HuggingFace…"):
            try:
                st.session_state.model = load_model(fname)
                st.session_state.model_name = model_choice
                st.success(f"✅ {fname} ready")
            except Exception as e:
                st.error(f"Load failed: {e}")

    if st.session_state.model_name:
        st.markdown(f'<span class="stat-badge">✓ {MODEL_FILES[st.session_state.model_name]}</span>', unsafe_allow_html=True)

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">② Images</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.images:
                st.session_state.images[f.name] = Image.open(f).convert("RGB")
                st.session_state.annotations.setdefault(f.name, [])
        if not st.session_state.current_img and st.session_state.images:
            st.session_state.current_img = list(st.session_state.images.keys())[0]

    total_imgs = len(st.session_state.images)
    annotated = sum(1 for v in st.session_state.annotations.values() if v)
    if total_imgs:
        st.markdown(
            f'<span class="stat-badge">{total_imgs} images</span>'
            f'<span class="stat-badge">{annotated} annotated</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Inference controls ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">③ Inference</div>', unsafe_allow_html=True)
    st.session_state.conf_thresh = st.slider(
        "Confidence threshold", 0.05, 0.95,
        st.session_state.conf_thresh, 0.05,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ This image"):
            cimg = st.session_state.current_img
            if cimg and st.session_state.model:
                with st.spinner("Running…"):
                    boxes = run_inference(
                        st.session_state.images[cimg],
                        st.session_state.conf_thresh,
                    )
                    st.session_state.annotations[cimg] = boxes
                    st.session_state.selected_box = None
            elif not st.session_state.model:
                st.warning("Load a model first.")
    with col2:
        if st.button("▶▶ All"):
            if st.session_state.model:
                prog = st.progress(0)
                imgs = list(st.session_state.images.items())
                for i, (fname, pil) in enumerate(imgs):
                    boxes = run_inference(pil, st.session_state.conf_thresh)
                    st.session_state.annotations[fname] = boxes
                    prog.progress((i + 1) / len(imgs))
                prog.empty()
                st.success("Done!")
            else:
                st.warning("Load a model first.")

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">④ Export</div>', unsafe_allow_html=True)
    if st.session_state.annotations:
        yolo_zip = export_yolo_zip()
        st.download_button(
            "⬇ YOLO dataset (.zip)",
            data=yolo_zip,
            file_name="yolo_dataset.zip",
            mime="application/zip",
        )
        coco_zip = export_coco_zip()
        st.download_button(
            "⬇ COCO dataset (.zip)",
            data=coco_zip,
            file_name="coco_dataset.zip",
            mime="application/zip",
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.images:
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #444; font-family: 'Space Mono', monospace;">
        <div style="font-size: 3rem; margin-bottom: 16px;">📂</div>
        <div style="font-size: 0.9rem; color: #666;">Upload images in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Image selector ────────────────────────────────────────────────────────────
img_names = list(st.session_state.images.keys())
col_sel, col_nav = st.columns([4, 1])
with col_sel:
    current = st.selectbox(
        "Current image",
        img_names,
        index=img_names.index(st.session_state.current_img) if st.session_state.current_img in img_names else 0,
        label_visibility="collapsed",
    )
    if current != st.session_state.current_img:
        st.session_state.current_img = current
        st.session_state.selected_box = None

with col_nav:
    cidx = img_names.index(st.session_state.current_img)
    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("◀") and cidx > 0:
            st.session_state.current_img = img_names[cidx - 1]
            st.session_state.selected_box = None
            st.rerun()
    with nc2:
        if st.button("▶") and cidx < len(img_names) - 1:
            st.session_state.current_img = img_names[cidx + 1]
            st.session_state.selected_box = None
            st.rerun()

# ── Main two-column layout ────────────────────────────────────────────────────
img_col, edit_col = st.columns([3, 1])

current_img = st.session_state.current_img
pil_img = st.session_state.images[current_img]
boxes = st.session_state.annotations.get(current_img, [])

with img_col:
    drawn = draw_boxes(pil_img, boxes, st.session_state.selected_box)
    st.image(drawn, use_container_width=True, caption=f"{current_img}  ({pil_img.width}×{pil_img.height})  —  {len(boxes)} box{'es' if len(boxes) != 1 else ''}")

with edit_col:
    st.markdown('<div class="sidebar-section">Boxes</div>', unsafe_allow_html=True)

    if not boxes:
        st.markdown('<p style="color:#555; font-size:0.8rem; font-family: Space Mono, monospace;">No annotations yet.<br>Run inference or add boxes.</p>', unsafe_allow_html=True)
    else:
        for i, box in enumerate(boxes):
            cid = box["class_id"]
            label = st.session_state.labels[cid] if cid < len(st.session_state.labels) else str(cid)
            conf = box.get("conf", 1.0)
            color = CLASS_PALETTE[cid % len(CLASS_PALETTE)]
            is_sel = (i == st.session_state.selected_box)
            sel_class = "selected" if is_sel else ""
            if st.button(
                f"{'▶ ' if is_sel else ''}#{i} {label} ({conf:.2f})",
                key=f"box_{i}",
                use_container_width=True,
            ):
                st.session_state.selected_box = i if not is_sel else None
                st.rerun()

    st.divider()

    # ── Edit selected box ─────────────────────────────────────────────────────
    sel = st.session_state.selected_box
    if sel is not None and sel < len(boxes):
        st.markdown('<div class="sidebar-section">Edit Box</div>', unsafe_allow_html=True)
        box = boxes[sel]

        new_class = st.selectbox(
            "Class",
            range(len(st.session_state.labels)),
            index=box["class_id"],
            format_func=lambda i: f"{i}: {st.session_state.labels[i]}",
            key="edit_class",
        )

        new_cx = st.slider("Center X", 0.0, 1.0, float(box["cx"]), 0.005, key="edit_cx")
        new_cy = st.slider("Center Y", 0.0, 1.0, float(box["cy"]), 0.005, key="edit_cy")
        new_w  = st.slider("Width",    0.01, 1.0, float(box["w"]),  0.005, key="edit_w")
        new_h  = st.slider("Height",   0.01, 1.0, float(box["h"]),  0.005, key="edit_h")

        if st.button("💾 Apply", use_container_width=True):
            boxes[sel].update({
                "class_id": new_class,
                "cx": new_cx, "cy": new_cy,
                "w": new_w,   "h": new_h,
            })
            st.session_state.annotations[current_img] = boxes
            st.rerun()

        if st.button("🗑 Delete box", use_container_width=True):
            boxes.pop(sel)
            st.session_state.annotations[current_img] = boxes
            st.session_state.selected_box = None
            st.rerun()

    st.divider()

    # ── Add new box manually ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Add Box</div>', unsafe_allow_html=True)
    add_class = st.selectbox(
        "Class",
        range(len(st.session_state.labels)),
        format_func=lambda i: f"{i}: {st.session_state.labels[i]}",
        key="add_class",
    )
    add_cx = st.slider("Center X", 0.0, 1.0, 0.5, 0.005, key="add_cx")
    add_cy = st.slider("Center Y", 0.0, 1.0, 0.5, 0.005, key="add_cy")
    add_w  = st.slider("Width",    0.05, 1.0, 0.2, 0.005, key="add_w")
    add_h  = st.slider("Height",   0.05, 1.0, 0.2, 0.005, key="add_h")

    if st.button("➕ Add box", use_container_width=True):
        boxes.append({
            "class_id": add_class,
            "cx": add_cx, "cy": add_cy,
            "w": add_w,   "h": add_h,
            "conf": 1.0,
        })
        st.session_state.annotations[current_img] = boxes
        st.session_state.selected_box = len(boxes) - 1
        st.rerun()

    # ── Clear current image ───────────────────────────────────────────────────
    st.divider()
    if st.button("🧹 Clear all boxes", use_container_width=True):
        st.session_state.annotations[current_img] = []
        st.session_state.selected_box = None
        st.rerun()
