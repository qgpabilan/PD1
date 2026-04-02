"""
PCB Solder Defect Detection V2 — Train from SolDef_AI
CPEQC-029-030 | Team 4 | TIP Quezon City

SolDef_AI Structure detected:
  Labeled/   → 428 images + 428 JSON annotations  ← PRIMARY SOURCE
  Dataset/   → CS1..CS7 folders with V1(good)/V2/V3(defect) images

Class mapping from SolDef_AI JSON tags:
  No_Defect          → 0
  Solder_Bridge      → 1
  Insufficient_Solder→ 2
  Excess_Solder      → 3
  Solder_Spike       → 4
"""
import json, shutil, os, random
from pathlib import Path
from multiprocessing import freeze_support

# ── CONFIG ────────────────────────────────────────────────
PROJECT_DIR  = Path(r"C:\Users\Gabriel Pabilan\Downloads\PCB_V2")
SOLDEF_DIR   = PROJECT_DIR / "SolDef_AI"
LABELED_DIR  = SOLDEF_DIR / "Labeled"
DATASET_DIR  = PROJECT_DIR / "dataset"
DATASET_DIR_RAW = PROJECT_DIR / "Dataset"

EPOCHS     = 25
IMG_SIZE   = 640
TRAIN_RATIO= 0.75
VAL_RATIO  = 0.15
# TEST = remaining 0.10

CLASS_NAMES = {
    0: "No_Defect",
    1: "Solder_Bridge",
    2: "Insufficient_Solder",
    3: "Excess_Solder",
    4: "Solder_Spike",
}

# SolDef_AI JSON tag → your class ID
# Based on the paper: tags are "No_Defect","Excess_Solder",
# "Insufficient_Solder","Solder_Bridge","Spike"
TAG_MAP = {
    # No Defect
    "no_defect":0, "no defect":0, "good":0, "correct":0,
    "ok":0, "normal":0, "pass":0, "v1":0,
    # Solder Bridge
    "solder_bridge":1, "solder bridge":1, "bridge":1,
    "bridging":1, "short":1,
    # Insufficient Solder
    "insufficient_solder":2, "insufficient solder":2,
    "insufficient":2, "less_solder":2, "poor":2,
    # Excess Solder
    "excess_solder":3, "excess solder":3, "excess":3,
    "excessive":3, "too_much":3,
    # Solder Spike / Cold
    "spike":4, "solder_spike":4, "solder spike":4,
    "spikes":4, "cold":4, "cold_solder":4,
}

# Folder version → class ID
# V1 = good/no defect, V2/V2.1/V3 = defects
FOLDER_VERSION_MAP = {
    "v1": 0,     # Good
    "v2": 3,     # Excess Solder (most common defect in dataset)
    "v2.1": 2,   # Insufficient Solder
    "v3": 4,     # Solder Spike
}

def match_tag(tag):
    t = tag.lower().strip().replace(" ","_")
    if t in TAG_MAP: return TAG_MAP[t]
    for kw, cid in TAG_MAP.items():
        if kw in t: return cid
    return None

def normalize_bbox(x, y, w, h, img_w, img_h):
    """Convert absolute bbox to YOLO normalized format"""
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

def process_labeled_folder():
    """Process Labeled/ folder with JSON annotations"""
    print(f"\n  Processing Labeled/ folder ({LABELED_DIR})")
    
    json_files = list(LABELED_DIR.glob("*.json"))
    img_files  = list(LABELED_DIR.glob("*.jpg"))
    
    print(f"  JSON files : {len(json_files)}")
    print(f"  JPG files  : {len(img_files)}")
    
    if not json_files:
        print("  No JSON files found in Labeled/")
        return []
    
    # Read first JSON to understand format
    sample = json.loads(json_files[0].read_text(encoding="utf-8"))
    print(f"\n  Sample JSON keys: {list(sample.keys())}")
    
    pairs = []
    class_counts = {i: 0 for i in range(5)}
    skipped = 0
    
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8", errors="ignore"))
            
            # Find matching image
            img_path = LABELED_DIR / json_file.with_suffix(".jpg").name
            if not img_path.exists():
                # Try other extensions
                for ext in [".jpeg", ".png", ".JPG"]:
                    alt = LABELED_DIR / json_file.with_suffix(ext).name
                    if alt.exists():
                        img_path = alt
                        break
            
            if not img_path.exists():
                skipped += 1
                continue
            
            # Get image dimensions
            img_w = data.get("imageWidth", data.get("image_width", 640))
            img_h = data.get("imageHeight", data.get("image_height", 640))
            
            # Handle different JSON formats
            yolo_lines = []
            
            # Format 1: LabelMe format {"shapes":[{"label":..,"points":...}]}
            if "shapes" in data:
                for shape in data["shapes"]:
                    label  = shape.get("label","").strip()
                    cls_id = match_tag(label)
                    if cls_id is None:
                        cls_id = 0  # default to No_Defect
                    
                    pts = shape.get("points", [])
                    if not pts: continue
                    
                    shape_type = shape.get("shape_type", "rectangle")
                    
                    if shape_type == "rectangle" and len(pts) == 2:
                        x1,y1 = pts[0]; x2,y2 = pts[1]
                        x,y   = min(x1,x2), min(y1,y2)
                        w,h   = abs(x2-x1), abs(y2-y1)
                    elif shape_type == "polygon" or len(pts) > 2:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x,y = min(xs),min(ys)
                        w,h = max(xs)-x, max(ys)-y
                    else:
                        continue
                    
                    if w <= 0 or h <= 0: continue
                    cx,cy,nw,nh = normalize_bbox(x,y,w,h,img_w,img_h)
                    cx = max(0,min(1,cx)); cy = max(0,min(1,cy))
                    nw = max(0,min(1,nw)); nh = max(0,min(1,nh))
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    class_counts[cls_id] += 1
            
            # Format 2: COCO-style {"annotations":[{"category_id":..,"bbox":[x,y,w,h]}]}
            elif "annotations" in data:
                cats = {c["id"]:c["name"] for c in data.get("categories",[])}
                for ann in data["annotations"]:
                    cat_name = cats.get(ann.get("category_id",""),"")
                    cls_id   = match_tag(cat_name)
                    if cls_id is None: cls_id = 0
                    bbox = ann.get("bbox",[])
                    if len(bbox) == 4:
                        x,y,w,h = bbox
                        cx,cy,nw,nh = normalize_bbox(x,y,w,h,img_w,img_h)
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        class_counts[cls_id] += 1
            
            # Format 3: Custom {"defects":[{"type":..,"bbox":{...}}]}
            elif "defects" in data:
                for defect in data["defects"]:
                    dtype  = defect.get("type","")
                    cls_id = match_tag(dtype)
                    if cls_id is None: cls_id = 0
                    bbox = defect.get("bbox",{})
                    if bbox:
                        x = bbox.get("x",0); y = bbox.get("y",0)
                        w = bbox.get("w",bbox.get("width",0))
                        h = bbox.get("h",bbox.get("height",0))
                        cx,cy,nw,nh = normalize_bbox(x,y,w,h,img_w,img_h)
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        class_counts[cls_id] += 1
            
            # If no annotations found, treat as No_Defect
            if not yolo_lines:
                yolo_lines = [f"0 0.500000 0.500000 0.900000 0.900000"]
                class_counts[0] += 1
            
            pairs.append((img_path, yolo_lines))
        
        except Exception as e:
            skipped += 1
    
    print(f"  Processed  : {len(pairs)} images")
    print(f"  Skipped    : {skipped}")
    print(f"  Classes from Labeled/:")
    for cid, count in class_counts.items():
        print(f"    [{cid}] {CLASS_NAMES[cid]:25s}: {count}")
    
    return pairs

def process_dataset_folder():
    """Process Dataset/CS*/R*/V* folder structure"""
    print(f"\n  Processing Dataset/ folder structure...")
    
    dataset_base = SOLDEF_DIR / "Dataset"
    if not dataset_base.exists():
        print("  Dataset/ not found — skipping")
        return [], {}
    
    pairs = []
    class_counts = {i: 0 for i in range(5)}
    
    # Walk all image files
    for img_file in dataset_base.rglob("*.jpg"):
        # Determine class from folder version
        parts = [p.lower() for p in img_file.parts]
        
        cls_id = None
        for part in reversed(parts):
            if part in FOLDER_VERSION_MAP:
                cls_id = FOLDER_VERSION_MAP[part]
                break
        
        if cls_id is None:
            # Check if "setup1" or "setup2" → V1 = good
            if "setup1" in parts or "setup2" in parts:
                cls_id = 0  # No_Defect
            else:
                cls_id = 0  # Default

        # Full image bbox label
        yolo_line = f"{cls_id} 0.500000 0.500000 0.900000 0.900000"
        pairs.append((img_file, [yolo_line]))
        class_counts[cls_id] += 1
    
    print(f"  Found: {len(pairs)} images from Dataset/")
    print(f"  Classes from Dataset/:")
    for cid, count in class_counts.items():
        if count > 0:
            print(f"    [{cid}] {CLASS_NAMES[cid]:25s}: {count}")
    
    return pairs, class_counts

def build_yolo_dataset(all_pairs):
    """Split pairs into train/val/test and save"""
    print(f"\n  Building YOLO dataset from {len(all_pairs)} images...")
    
    # Create folders
    for split in ["train","val","test"]:
        (DATASET_DIR/split/"images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR/split/"labels").mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_pairs)
    n       = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    
    splits = (
        [("train", p) for p in all_pairs[:n_train]] +
        [("val",   p) for p in all_pairs[n_train:n_train+n_val]] +
        [("test",  p) for p in all_pairs[n_train+n_val:]]
    )
    
    class_counts = {i: 0 for i in range(5)}
    saved = {"train":0, "val":0, "test":0}
    
    for split, (img_path, yolo_lines) in splits:
        # Unique filename
        dst_img = DATASET_DIR/split/"images"/img_path.name
        dst_lbl = DATASET_DIR/split/"labels"/img_path.with_suffix(".txt").name
        counter = 0
        while dst_img.exists():
            counter += 1
            dst_img = DATASET_DIR/split/"images"/f"{img_path.stem}_{counter}{img_path.suffix}"
            dst_lbl = DATASET_DIR/split/"labels"/f"{img_path.stem}_{counter}.txt"
        
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines))
        
        for line in yolo_lines:
            parts = line.strip().split()
            if parts:
                try: class_counts[int(parts[0])] += 1
                except: pass
        
        saved[split] += 1
    
    return saved, class_counts

if __name__ == '__main__':
    freeze_support()
    import torch
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count()>0 else "cpu"

    print("="*60)
    print("  PCB DEFECT DETECTION V2 — SolDef_AI TRAINING")
    print("  CPEQC-029-030 | Team 4 | TIP Quezon City")
    print(f"  Device  : {DEVICE.upper()}")
    print(f"  Epochs  : {EPOCHS}")
    print("="*60)

    # Clear old dataset
    if DATASET_DIR.exists():
        print(f"\n  Clearing old dataset...")
        shutil.rmtree(DATASET_DIR)

    # ── Collect all pairs ─────────────────────────────────
    all_pairs = []

    # 1. Labeled/ folder (428 images with JSON — BEST quality)
    labeled_pairs = process_labeled_folder()
    all_pairs.extend(labeled_pairs)

    # 2. Dataset/ folder (V1/V2/V3 images — adds volume)
    dataset_pairs, _ = process_dataset_folder()
    all_pairs.extend(dataset_pairs)

    print(f"\n  Total pairs collected: {len(all_pairs)}")

    if len(all_pairs) == 0:
        print("  ERROR: No images found! Check SolDef_AI folder.")
        exit(1)

    # ── Build YOLO dataset ────────────────────────────────
    saved, class_counts = build_yolo_dataset(all_pairs)

    # ── Write data.yaml ───────────────────────────────────
    yaml_out = DATASET_DIR / "data.yaml"
    yaml_out.write_text(
        f"train: {DATASET_DIR}/train/images\n"
        f"val:   {DATASET_DIR}/val/images\n"
        f"test:  {DATASET_DIR}/test/images\n\n"
        f"nc: 5\n"
        f"names: ['No_Defect','Solder_Bridge','Insufficient_Solder','Excess_Solder','Solder_Spike']\n"
    )

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    for split, count in saved.items():
        print(f"  {split:6s}: {count} images")
    print(f"\n  Per-class instance counts:")
    for cid, count in class_counts.items():
        bar  = "#"*min(count//3, 40)
        warn = "  <- needs more data!" if count < 50 else ""
        print(f"  [{cid}] {CLASS_NAMES[cid]:25s}: {count:5d} {bar}{warn}")
    print(f"\n  data.yaml: {yaml_out}")

    # ── Train ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING")
    print(f"  EPOCHS={EPOCHS} | DEVICE={DEVICE.upper()} | IMGSZ={IMG_SIZE}")
    print(f"{'='*60}")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(
        data      = str(yaml_out),
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = 16,
        device    = DEVICE,
        project   = str(PROJECT_DIR / "runs"),
        name      = "PCB_V2",
        exist_ok  = True,
        optimizer = "AdamW",
        lr0       = 0.001,
        plots     = True,
        patience  = 20,
        workers   = 0,
        verbose   = True,
    )

    print("\n✅ Training complete!")
    print(f"  Weights: {PROJECT_DIR}\\runs\\PCB_V2\\weights\\best.pt")
    print(f"\n  Next step — run the app:")
    print(f"  python app_v3.py")
