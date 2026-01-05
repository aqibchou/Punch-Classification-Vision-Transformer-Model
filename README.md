#  Combat Sports AI Video Analysis

A Vision-Transformer Based application for analyzing combat sports by classifying boxing actions in real-time.

##  Features

###  Core Analysis
- **Action Classification**: Automatic classification of 8 boxing action types:
  - Head hits (left/right)
  - Body hits (left/right)
  - Blocks (left/right)
  - Misses (left/right)
- **Vision Transformer Model**: State-of-the-art transformer architecture for video action recognition
- **Dense Prediction**: Per-frame classification to detect multiple actions within sequences
- **Peak Detection**: Advanced algorithm to identify distinct actions, including rapid combos

###  Advanced Features
- **Real-time Inference**: Fast video processing done with GPU acceleration through a VM on Google Cloud
- **Sliding Window Analysis**: Processes videos with overlapping windows for comprehensive coverage
- **Confidence Scoring**: Action detection with confidence thresholds
- **Class Balancing**: Automatic handling of class imbalance in training data

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- NumPy, Pandas, Scikit-learn
- tqdm

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd "Combat Sports Video Analysis"

# Install dependencies
pip install -r requirements.txt
```

##  Usage

### Inference on Video
```bash
# Run inference on a video file
python scripts/inference_video_transformer.py \
    --checkpoint models/best_model_dense_finetuned.pth \
    --video path/to/video.mp4 \
    --output results.json \
    --device cuda \
    --min-peak-height 0.15 \
    --min-peak-gap 2 \
    --boost-under-detected
```

### Training
```bash
# Prepare training data
python data_preparation/prepare_video_training_data.py \
    --converted-dir converted_annotations \
    --video-dir training_data \
    --output-dir training_data_video \
    --no-action-file no_action_sequences.json

# Pre-extract frames (recommended for faster training)
python data_preparation/pre_extract_frames.py \
    --train-file training_data_video/train.json \
    --val-file training_data_video/val.json \
    --video-dir training_data \
    --output-dir extracted_frames

# Train the model
python scripts/train_transformer_video.py \
    --train-file training_data_video/train.json \
    --val-file training_data_video/val.json \
    --extracted-frames-dir extracted_frames \
    --output-dir models \
    --epochs 50 \
    --batch-size 8 \
    --dense-prediction
```

## 📁 Project Structure

```
Combat Sports Video Analysis/
├── models/                                    # Model architecture files
│   ├── actionnet_transformer_video.py        # Vision Transformer model
│   ├── actionnet_transformer.py             # Transformer base classes
│   └── adaptok_tokenizer.py                  # Adaptive tokenization
├── scripts/                                   # Training and inference scripts
│   ├── train_transformer_video.py            # Main training script
│   ├── fine_tune_dense.py                    # Fine-tuning script
│   ├── inference_video_transformer.py        # Inference script
│   └── restart_training.sh                   # Training helper script
├── data_preparation/                          # Data preparation utilities
│   ├── prepare_video_training_data.py         # Create training/val splits
│   ├── pre_extract_frames.py                 # Extract frames for training
│   └── extract_no_action_sequences.py        # Extract no-action sequences
├── data/                                      # Data directory (not in repo)
│   └── sample_videos/                        # Training videos
├── README.md                                  # This file
└── requirements.txt                           # Python dependencies
```

##  Model Architecture

### Vision Transformer for Video
- **Input**: Raw RGB video frames (224x224, 30 frames per sequence)
- **Architecture**: Transformer encoder with patch embedding
- **Output**: Dense per-frame predictions (30 frames × 9 classes)
- **Classes**: 8 action types + 1 "no_action" class

### Key Components
1. **Frame Patch Embedding**: Converts video frames into patch tokens
2. **Positional Encoding**: Adds temporal position information
3. **Transformer Encoder**: Multi-head self-attention for sequence modeling
4. **Dense Classification Head**: Per-frame action predictions
5. **Peak Detection**: Post-processing to identify distinct actions

##  Training Data

The model is trained on the Olympic Boxing Punch Classification Video Dataset, which includes:
- Frame-by-frame annotations by licensed boxing referees
- 8 action classes: head/body hits, blocks, misses (each left/right)
- High-quality boxing footage with precise timing

### Dataset Citation
```bibtex
@article{stefanski2024boxing,
  title={Boxing Punch Detection with Single Static Camera},
  author={Stefański, Piotr and Kozak, Jan and Jach, Tomasz},
  journal={Entropy},
  volume={26},
  number={8},
  year={2024},
  publisher={Multidisciplinary Digital Publishing Institute (MDPI)}
}
```

##  Technical Details

### Model Performance
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~91%
- **Architecture**: Vision Transformer with 3 encoder layers, 8 attention heads
- **Input**: 30-frame sequences at 224x224 resolution
- **Output**: Dense predictions for all frames

### Inference
- **Processing Speed**: ~1-1.5 seconds per 30-frame sequence (on GPU)
- **Detection Method**: Peak detection with configurable thresholds
- **Class Balancing**: Confidence boosts for under-detected classes

### Supported Video Formats
- MP4, MOV, AVI
- Any resolution (automatically resized to 224x224)
- Frame rate: 24-60 FPS recommended

##  Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run inference on a video**:
   ```bash
   python scripts/inference_video_transformer.py \
       --checkpoint models/best_model_dense_finetuned.pth \
       --video your_video.mp4 \
       --output results.json
   ```

3. **View results**:
   The output JSON file contains:
   - Action counts by type
   - Detection timestamps
   - Confidence scores
   - Statistics (hit rate, block rate, miss rate)

##  Development Status

- [x] ✅ Vision Transformer model implementation
- [x] ✅ Dense prediction mode
- [x] ✅ Peak detection algorithm
- [x] ✅ Training pipeline
- [x] ✅ Inference pipeline
- [x] ✅ Class balancing and confidence boosting
- [x] ✅ Frame pre-extraction for faster training

##  Contributing

Contributions are welcome! Areas for improvement:
- Model accuracy improvements
- Additional action types
- Seperating actions for both opponents
- Using the Seperate Actions to create an unbiased round by round judging system.
- Real-time processing optimization
- Better handling of class imbalance

##  License

MIT License - see LICENSE file for details.

##  Acknowledgments

This project uses the Olympic Boxing Punch Classification Video Dataset:
- **Citation**: Stefański, P., Kozak, J., & Jach, T. (2024). Boxing Punch Detection with Single Static Camera. *Entropy*, 26(8).


