"""
Baseline YOLOv8n Training Script
==================================
Trains a standard YOLOv8n model for PPE detection to establish baseline metrics.
Paper reports: mAP@0.5 = 70.2%, mAP@0.5:0.95 = 35.1%, Params = 3.0M, GFLOPs = 8.1

Usage:
    cd /home/surya/miniproject_antigravity
    ./venv/bin/python train_baseline.py
"""

import sys
sys.path.insert(0, '/home/surya/miniproject_antigravity')

from ultralytics import YOLO


def main():
    # Load YOLOv8n model (from scratch, no pretrained weights — matches paper)
    model = YOLO('yolov8n.yaml')

    # Training hyperparameters (matching paper exactly, adjusted for hardware)
    results = model.train(
        data='/home/surya/miniproject_antigravity/dataset.yaml',
        epochs=300,
        batch=16,           # Matches paper
        imgsz=640,
        lr0=0.01,           # Ultralytics default; final LR = lr0*lrf = 0.0001
        lrf=0.01,           # Final LR = lr0 * lrf
        weight_decay=0.0005,
        optimizer='SGD',
        warmup_epochs=1.0,  # ~200 iterations warmup
        pretrained=False,   # Train from scratch like the paper
        amp=True,           # Mixed precision for VRAM efficiency
        project='runs/baseline',
        name='final',
        save_period=50,     # Save checkpoint every 50 epochs
        patience=0,         # No early stopping — full 300 epochs
        verbose=True,
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("Baseline YOLOv8n — Test Set Evaluation")
    print("="*60)
    
    metrics = model.val(
        data='/home/surya/miniproject_antigravity/dataset.yaml',
        split='test',
    )
    
    print(f"mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Paper target: mAP@0.5=0.702, mAP@0.5:0.95=0.351")


if __name__ == '__main__':
    main()
