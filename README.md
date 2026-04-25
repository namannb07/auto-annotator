# Auto-Annotator — Streamlit Web App

YOLOv8 (Ultralytics/YOLOv8 on HuggingFace) · CPU inference · YOLO & COCO export

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

1. **Load Model** — pick a YOLOv8 variant in the sidebar and click "Load Model".
   Downloads from `Ultralytics/YOLOv8` on HuggingFace (cached after first run).

2. **Upload Images** — drag & drop JPG/PNG files via the sidebar uploader.

3. **Run Inference**
   - `▶ This image` — annotate the current image
   - `▶▶ All` — annotate every uploaded image in batch

4. **Edit Annotations**
   - Click a box in the **Boxes** list to select it
   - Adjust class, center X/Y, width, height with sliders
   - Hit **Apply** to save edits, or **Delete** to remove
   - Use **Add Box** sliders to place a new box manually

5. **Export**
   - `⬇ YOLO dataset` — ZIP with `images/`, `labels/`, `classes.txt`, `data.yaml`
   - `⬇ COCO dataset` — ZIP with `images/`, `annotations.json`

---

## Deploying to Hugging Face Spaces (free)

1. Create a new Space on huggingface.co (SDK: Streamlit)
2. Upload `app.py` and `requirements.txt`
3. Space builds automatically — no GPU needed

---

## Model variants

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| YOLOv8n | 6 MB | Fastest | Lowest |
| YOLOv8s | 22 MB | Fast | Good |
| YOLOv8m | 52 MB | Medium | Better |
| YOLOv8l | 88 MB | Slower | Best |

All run on CPU. YOLOv8n is recommended for free-tier cloud (512 MB RAM limit).
YOLOv8s/m work well on HuggingFace Spaces (16 GB RAM).
