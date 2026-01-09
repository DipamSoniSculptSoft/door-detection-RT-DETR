import os
import json
import numpy as np
from PIL import Image
import random
import yaml
import sys

# =================================================================
# CONFIGURATION
# =================================================================
DATA_DIR = os.path.abspath("inhouse_data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
ANN_DIR = os.path.join(DATA_DIR, "annotations")

PROJECT_DIR = os.path.abspath("rtdetrv2_pytorch")
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")

# Training hyperparams
NUM_CLASSES = 1 # Just 'door'
CLASS_NAME = "door"

def get_bbox_from_mask(mask_path):
    """Extracts a bounding box from a binary mask image."""
    try:
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        
        # Find pixels where mask is non-zero
        pos = np.where(mask_np > 10) 
        if len(pos[0]) == 0:
            return None
        
        ymin, xmin = np.min(pos[0]), np.min(pos[1])
        ymax, xmax = np.max(pos[0]), np.max(pos[1])
        
        # COCO format: [x, y, width, height]
        return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None

def prepare_data():
    """Converts masks to COCO JSON and splits into train/val sets."""
    print("--- Preparing Data ---")
    if not os.path.exists(ANN_DIR):
        os.makedirs(ANN_DIR)

    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_files)} image files in {IMAGES_DIR}.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def create_coco(files, output_name):
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": CLASS_NAME}]
        }
        
        ann_id = 1
        img_count = 0
        for i, filename in enumerate(files):
            img_path = os.path.join(IMAGES_DIR, filename)
            base = os.path.splitext(filename)[0]
            
            # Try to find a matching mask
            mask_path = None
            potential_mask_names = [filename, base + ".png", base + ".PNG", base + ".jpg", base + ".jpeg"]
            for pmn in potential_mask_names:
                pmp = os.path.join(MASKS_DIR, pmn)
                if os.path.exists(pmp):
                    mask_path = pmp
                    break

            if mask_path is None:
                continue

            try:
                img = Image.open(img_path)
                w, h = img.size
                
                bbox = get_bbox_from_mask(mask_path)
                if bbox is None:
                    continue

                coco["images"].append({
                    "id": i+1,
                    "file_name": filename,
                    "width": w,
                    "height": h
                })
                
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": i+1,
                    "category_id": 0,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": [] 
                })
                ann_id += 1
                img_count += 1
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

        output_path = os.path.join(ANN_DIR, output_name)
        with open(output_path, 'w') as f:
            json.dump(coco, f)
        print(f"Saved {img_count} images to {output_path}")

    create_coco(train_files, "instances_train.json")
    create_coco(val_files, "instances_val.json")

def create_dataset_config():
    """Only creates the dataset YAML configuration file."""
    print("--- Updating Dataset Config ---")
    
    # Dataset Config
    dataset_cfg_path = os.path.join(CONFIGS_DIR, "dataset", "door_detection.yml")
    
    # We use a custom dumper to use ~ for None values if needed, 
    # but RT-DETR works fine with null/empty. 
    # Using None will result in 'null' in the YAML.
    dataset_cfg = {
        'task': 'detection',
        'evaluator': {
            'type': 'CocoEvaluator',
            'iou_types': ['bbox']
        },
        'num_classes': NUM_CLASSES,
        'remap_mscoco_category': False,
        'train_dataloader': {
            'type': 'DataLoader',
            'dataset': {
                'type': 'CocoDetection',
                'img_folder': IMAGES_DIR,
                'ann_file': os.path.join(ANN_DIR, "instances_train.json"),
                'return_masks': False,
                'transforms': {
                    'type': 'Compose',
                    'ops': None 
                }
            },
            'shuffle': True,
            'num_workers': 4,
            'drop_last': True,
            'collate_fn': {
                'type': 'BatchImageCollateFunction'
            }
        },
        'val_dataloader': {
            'type': 'DataLoader',
            'dataset': {
                'type': 'CocoDetection',
                'img_folder': IMAGES_DIR,
                'ann_file': os.path.join(ANN_DIR, "instances_val.json"),
                'return_masks': False,
                'transforms': {
                    'type': 'Compose',
                    'ops': None
                }
            },
            'shuffle': False,
            'num_workers': 4,
            'drop_last': False,
            'collate_fn': {
                'type': 'BatchImageCollateFunction'
            }
        }
    }
    
    with open(dataset_cfg_path, 'w') as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)
    
    print(f"Updated: {dataset_cfg_path}")

def instruct_and_run():
    """Prints instructions for training using the user's manual model config."""
    print("\n" + "="*50)
    print("DATA PREPARED FOR TRAINING!")
    print("="*50)
    print(f"1. COCO annotations refreshed in {ANN_DIR}")
    print(f"2. Dataset config updated at: rtdetrv2_pytorch/configs/dataset/door_detection.yml")
    print("\nTraining will use your manual configuration at:")
    print("rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_door.yml")
    print("\nTo start training, run:")
    print(f"python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_door.yml --use-amp")
    print("\nIMPORTANT: Please check the following in your manual configs:")
    print("- Ensure 'rtdetrv2_r18vd.yml' actually exists or use 'rtdetrv2_r50vd.yml' as base.")
    print("- Verify that custom transforms like 'CLAHEEnhance' are implemented in the code.")
    print("="*50)

if __name__ == "__main__":
    if not os.path.exists("rtdetrv2_pytorch"):
        print("Error: Please run this script from the project root.")
        sys.exit(1)
        
    prepare_data()
    create_dataset_config()
    instruct_and_run()