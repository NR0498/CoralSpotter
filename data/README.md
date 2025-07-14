# Coral Reef Detection Dataset

This dataset contains synthetic coral reef images designed for training computer vision models to detect and classify coral reef lifeforms.

## Dataset Structure

```
data/
├── images/
│   ├── train/          # 32 YOLO training images (640x640px)
│   └── val/            # 8 YOLO validation images (640x640px)
├── labels/
│   ├── train/          # YOLO format annotations (.txt files)
│   └── val/            # YOLO format validation annotations
├── coral_types/        # Classification training data
│   ├── brain_coral/    # 15 brain coral images
│   ├── staghorn_coral/ # 15 staghorn coral images
│   ├── table_coral/    # 15 table coral images
│   ├── soft_coral/     # 15 soft coral images
│   ├── fan_coral/      # 15 fan coral images
│   └── mushroom_coral/ # 15 mushroom coral images
└── dataset_info.json   # Dataset metadata and statistics
```

## Dataset Statistics

- **Total Images**: 130
- **YOLO Detection**: 40 annotated images with bounding boxes
- **Classification**: 90 images across 6 coral species
- **Image Format**: JPG, 640x640 pixels for YOLO, variable sizes for classification
- **Annotation Format**: YOLO format (normalized coordinates)

## Coral Types Included

1. **Brain Coral**: Round corals with wavy, brain-like texture
2. **Staghorn Coral**: Branching corals with antler-like structures
3. **Table Coral**: Flat, plate-like coral formations
4. **Soft Coral**: Flexible, flowing coral structures
5. **Fan Coral**: Fan-shaped coral formations
6. **Mushroom Coral**: Dome-shaped, mushroom-like corals

## Data Generation

The dataset was synthetically generated using computer vision techniques to create realistic underwater coral reef scenes with:

- Underwater blue-green backgrounds with light ray effects
- Diverse coral shapes and colors based on real coral characteristics
- Proper YOLO annotation format with normalized bounding boxes
- Multiple coral instances per image for detection training

## Usage

### For YOLO Training
- Use `images/train/` and `labels/train/` for training
- Use `images/val/` and `labels/val/` for validation
- Dataset configuration file: `coral_dataset.yaml`

### For Classification Training
- Each subdirectory in `coral_types/` contains images of one coral species
- Images are suitable for CNN or machine learning classification models
- Recommended split: 80% training, 20% validation

## Model Performance

### Classification Model (Random Forest)
- **Accuracy**: 72.22% on validation set
- **Features**: Computer vision features including color statistics, texture, and edge density
- **Classes**: 6 coral types with balanced distribution

### Detection Model (YOLO)
- **Status**: Training in progress
- **Target**: Single class detection (coral vs. non-coral)
- **Annotations**: Normalized bounding box coordinates

## File Formats

### YOLO Annotations
```
# Format: class_id center_x center_y width height (all normalized 0-1)
0 0.5 0.3 0.2 0.4
0 0.7 0.6 0.15 0.25
```

### Dataset Info JSON
Contains comprehensive metadata including:
- Total image counts
- Training/validation splits
- Coral type distributions
- Dataset creation details

## Applications

This dataset is designed for:
- Coral reef monitoring and conservation
- Marine biology research
- Underwater computer vision applications
- Educational coral identification systems
- Environmental impact assessment

## License and Usage

This synthetic dataset is provided for educational and research purposes. The generated images simulate real coral reef characteristics but are artificially created for training purposes.