import os
import shutil
import json
import cv2
import numpy as np
import requests
import zipfile
import random
from pathlib import Path
from tqdm.auto import tqdm

# ==========================================
# 1. СКАЧИВАНИЕ И СКЛЕИВАНИЕ АРХИВА
# ==========================================
VERSION_TAG = "Dataset"
BASE_URL = f"https://github.com/SergKurchev/strawberry_synthetic_dataset/releases/download/{VERSION_TAG}"
FILES_TO_DOWNLOAD = [
    "strawberry_dataset.zip.001",
    "strawberry_dataset.zip.002",
    "strawberry_dataset.zip.003"
]
OUTPUT_ZIP = "strawberry_dataset.zip"

# Очистка старых данных перед стартом
if os.path.exists("strawberry_dataset"):
    shutil.rmtree("strawberry_dataset")
    print("🗑️ Старая папка удалена.")

print("⬇️ Скачиваем датасет...")
os.makedirs("temp_download", exist_ok=True)
with open(OUTPUT_ZIP, 'wb') as outfile:
    for filename in FILES_TO_DOWNLOAD:
        file_path = Path("temp_download") / filename
        url = f"{BASE_URL}/{filename}"
        print(f"   Скачивается {filename}...")
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"Ошибка скачивания {filename}")
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with open(file_path, 'rb') as infile:
            shutil.copyfileobj(infile, outfile)

print("📦 Распаковка архива...")
with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
    zip_ref.extractall(".")
shutil.rmtree("temp_download", ignore_errors=True)

# Фикс слешей для Linux (на всякий случай)
for filename in os.listdir("."):
    if "\\" in filename:
        new_path = filename.replace("\\", "/")
        parent = os.path.dirname(new_path)
        if parent: os.makedirs(parent, exist_ok=True)
        shutil.move(filename, new_path)

# ==========================================
# 2. ГЕНЕРАЦИЯ ПОЛИГОНОВ ИЗ МАСОК И JSON
# ==========================================
DATASET_PATH = Path("strawberry_dataset")
MASKS_DIR = DATASET_PATH / "masks"
JSON_PATH = DATASET_PATH / "annotations.json"
LABELS_DIR = DATASET_PATH / "labels"

if LABELS_DIR.exists():
    shutil.rmtree(LABELS_DIR)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

print("📖 Читаем COCO annotations.json...")
with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

image_annotations = {}
img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

for ann in coco_data['annotations']:
    img_name = img_id_to_name[ann['image_id']]
    if img_name not in image_annotations:
        image_annotations[img_name] = []
    image_annotations[img_name].append({
        'category_id': ann['category_id'],
        'color': ann['segmentation_color']
    })

print("🔄 Конвертируем маски в полигоны...")
for img_name, objects in tqdm(image_annotations.items(), desc="Генерация YOLO-разметки"):
    mask_path = MASKS_DIR / img_name
    txt_name = mask_path.stem + ".txt"
    txt_path = LABELS_DIR / txt_name
    
    if not mask_path.exists(): continue
        
    mask_bgr = cv2.imread(str(mask_path))
    if mask_bgr is None: continue
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    
    H, W = mask_rgb.shape[:2]
    yolo_lines = []
    
    for obj in objects:
        cat_id = obj['category_id']
        if cat_id == 3: continue # Пропускаем хвостики
            
        r, g, b = obj['color']
        target_color = np.array([r, g, b], dtype=np.uint8)
        binary_mask = cv2.inRange(mask_rgb, target_color, target_color)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 20: continue
            polygon = []
            for point in contour:
                x = point[0][0] / W
                y = point[0][1] / H
                polygon.append(f"{x:.6f} {y:.6f}")
                
            if len(polygon) > 2:
                yolo_lines.append(f"{cat_id} " + " ".join(polygon) + "\n")
                
    if yolo_lines: 
        with open(txt_path, 'w') as f:
            f.writelines(yolo_lines)

# ==========================================
# 3. РАЗДЕЛЕНИЕ НА TRAIN И VAL
# ==========================================
print("🔄 Разделение на train/val...")
images_dir = DATASET_PATH / "images"

valid_images = []
for img_path in images_dir.glob("*.png"):
    lbl_name = img_path.stem + ".txt"
    if (LABELS_DIR / lbl_name).exists() and os.path.getsize(LABELS_DIR / lbl_name) > 0:
        valid_images.append(img_path)

print(f"Найдено {len(valid_images)} валидных картинок с полигонами.")
random.shuffle(valid_images)

split_idx = int(len(valid_images) * 0.8)
train_imgs = valid_images[:split_idx]
val_imgs = valid_images[split_idx:]

for split in ['train', 'val']:
    (images_dir / split).mkdir(exist_ok=True, parents=True)
    (LABELS_DIR / split).mkdir(exist_ok=True, parents=True)

for img_path in train_imgs:
    shutil.move(str(img_path), str(images_dir / "train" / img_path.name))
    lbl_name = img_path.stem + ".txt"
    shutil.move(str(LABELS_DIR / lbl_name), str(LABELS_DIR / "train" / lbl_name))
        
for img_path in val_imgs:
    shutil.move(str(img_path), str(images_dir / "val" / img_path.name))
    lbl_name = img_path.stem + ".txt"
    shutil.move(str(LABELS_DIR / lbl_name), str(LABELS_DIR / "val" / lbl_name))

print("✅ Датасет полностью готов!")

DATASET_PATH = Path("strawberry_dataset")
MASKS_DIR = DATASET_PATH / "masks"
JSON_PATH = DATASET_PATH / "annotations.json"
LABELS_DIR = DATASET_PATH / "labels"

# 1. Удаляем сломанный кэш YOLO
for cache_file in LABELS_DIR.glob("**/*.cache"):
    os.remove(cache_file)

print("📖 Читаем COCO annotations.json...")
with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

image_annotations = {}
img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

for ann in coco_data['annotations']:
    img_name = img_id_to_name[ann['image_id']]
    if img_name not in image_annotations:
        image_annotations[img_name] = []
    image_annotations[img_name].append({
        'category_id': ann['category_id'],
        'color': ann['segmentation_color']
    })

print("⚡ Ультра-быстрое восстановление полигонов...")
fixed_count = 0

for img_name, objects in tqdm(image_annotations.items()):
    mask_path = MASKS_DIR / img_name
    if not mask_path.exists(): continue
        
    mask_bgr = cv2.imread(str(mask_path))
    if mask_bgr is None: continue
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    
    H, W = mask_rgb.shape[:2]
    yolo_lines = []
    
    # ---------------------------------------------------------
    # 🔥 ГЛАВНАЯ ОПТИМИЗАЦИЯ: Отсекаем шум!
    # ---------------------------------------------------------
    pixels = mask_rgb.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Берем только те цвета, которых больше 20 штук (не мусор) и которые не черные (не фон)
    valid_mask = (counts > 20) & ~np.all(unique_colors == 0, axis=1)
    valid_colors = unique_colors[valid_mask]
    
    for img_color in valid_colors:
        min_dist = float('inf')
        best_obj = None
        for obj in objects:
            json_color = np.array(obj['color'])
            dist = np.linalg.norm(img_color - json_color)
            if dist < min_dist:
                min_dist = dist
                best_obj = obj
        
        if best_obj is None or best_obj['category_id'] == 3:
            continue 
            
        cat_id = best_obj['category_id']
        
        binary_mask = cv2.inRange(mask_rgb, img_color, img_color)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 20: continue
            polygon = []
            for point in contour:
                x = point[0][0] / W
                y = point[0][1] / H
                polygon.append(f"{x:.6f} {y:.6f}")
            if len(polygon) > 2:
                yolo_lines.append(f"{cat_id} " + " ".join(polygon) + "\n")
                
    if yolo_lines:
        target_txt_train = LABELS_DIR / "train" / (mask_path.stem + ".txt")
        target_txt_val = LABELS_DIR / "val" / (mask_path.stem + ".txt")
        
        if (DATASET_PATH/"images"/"train"/img_name).exists():
            with open(target_txt_train, 'w') as f: f.writelines(yolo_lines)
            fixed_count += 1
        elif (DATASET_PATH/"images"/"val"/img_name).exists():
            with open(target_txt_val, 'w') as f: f.writelines(yolo_lines)
            fixed_count += 1

print(f"✅ Успешно восстановлено файлов: {fixed_count}. Можно обучать!")