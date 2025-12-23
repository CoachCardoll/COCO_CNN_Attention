# Image Captioning with CNN-RNN and Attention

This project implements an image captioning model using ResNet50 as the CNN encoder and an RNN decoder with attention mechanism, trained on the MS COCO 2014 dataset. **Supports pretrained model loading and fine-tuning!**

## Features

- **Encoder**: ResNet50 CNN for image feature extraction
- **Decoder**: LSTM-based RNN with Bahdanau attention mechanism
- **Dataset**: MS COCO 2014
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Attention Visualization**: Visualize what the model focuses on when generating captions
- **✨ Pretrained Model Support**: Load, save, and fine-tune pretrained models
- **Transfer Learning**: Adapt models for different datasets or vocabularies
- **🎯 Unified Utils**: All utilities consolidated in `utils.py` for easy access
- **Beam Search**: Advanced caption generation with beam search
- **📦 Batch Inference**: Process single images, multiple images, or entire directories
- **🎨 Attention Visualization**: See what the model focuses on for each word
- **Training Utilities**: Gradient clipping, learning rate adjustment, accuracy metrics

## Project Structure

```
.
├── config.py              # Configuration parameters
├── utils.py               # ⭐ All utility functions (vocabulary, pretrained, inference, etc.)
├── dataset.py            # COCO dataset loader
├── model.py              # Encoder and Decoder models
├── train.py              # Training from scratch
├── finetune.py           # Fine-tune pretrained models
├── export_pretrained.py  # Export trained models
├── inference.py          # Caption generation (single & batch)
├── requirements.txt      # Dependencies
├── README.md            # This file
├── UTILS_REFERENCE.md   # Complete utils.py documentation
└── INFERENCE_GUIDE.md   # Complete inference documentation
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download MS COCO 2014 dataset:
```bash
# Create data directory
mkdir -p data/coco2014

# Download and extract images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip -d data/coco2014/
unzip val2014.zip -d data/coco2014/

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d data/coco2014/
```

## Usage

### Option 1: Training from Scratch

Build vocabulary and train the model:
```bash
python train.py
```

The training script will:
- Build vocabulary from captions (if not already exists)
- Train the model with early stopping
- Save the best model checkpoint to `checkpoints/best_model.pt`

### Option 2: Using Pretrained Models

#### Load a Pretrained Model

```python
from utils import load_pretrained_model
from config import Config

config = Config()

# Load pretrained model
encoder, decoder, vocab = load_pretrained_model(
    pretrained_path='./pretrained_models/coco_captioning_pretrained.pt',
    vocab_path='./pretrained_models/vocab.pkl',
    device=config.DEVICE
)

# Now use for inference or fine-tuning
```

#### Fine-tune a Pretrained Model

Fine-tune on your own dataset:

```bash
# Fine-tune both encoder and decoder
python finetune.py \
    --pretrained_path ./pretrained_models/coco_captioning_pretrained.pt \
    --learning_rate 1e-4 \
    --epochs 10 \
    --output_name my_finetuned_model

# Fine-tune only decoder (freeze encoder)
python finetune.py \
    --pretrained_path ./pretrained_models/coco_captioning_pretrained.pt \
    --freeze_encoder \
    --learning_rate 1e-4 \
    --epochs 10 \
    --output_name my_finetuned_model
```

**Fine-tuning Tips:**
- Use lower learning rate (1e-4 or 1e-5) for fine-tuning
- Freeze encoder if your dataset is small to prevent overfitting
- Fewer epochs needed (5-15 usually sufficient)
- Fine-tuning is much faster than training from scratch

#### Export Your Trained Model

After training, export your model as a pretrained model for sharing:

```bash
python export_pretrained.py \
    --checkpoint ./checkpoints/best_model.pt \
    --vocab ./vocab.pkl \
    --output_dir ./pretrained_models \
    --model_name my_awesome_model
```

This creates a portable pretrained model that others can use!

### Inference

Generate captions for new images:

```bash
python inference.py
```

Update the `image_path` variable in `inference.py` to point to your test image.

## Pretrained Model Utilities

The `utils.py` module provides all necessary functions in one place:

```python
from utils import (
    # Vocabulary
    build_vocab, save_vocab, load_vocab, Vocabulary,
    
    # Pretrained Models
    load_pretrained_model, save_pretrained_model,
    load_checkpoint, adapt_vocab_size,
    
    # Model Utilities
    print_model_info, freeze_model, unfreeze_model,
    count_parameters, get_trainable_params,
    
    # Inference
    load_image, generate_caption, beam_search_caption,
    visualize_attention,
    
    # Training
    EarlyStopping, clip_gradients, adjust_learning_rate, accuracy
)
```

**See `UTILS_REFERENCE.md` for complete documentation with examples!**

## Model Architecture

### Encoder (CNN)
- **Base**: ResNet50 pretrained on ImageNet
- **Output**: 14×14×2048 feature maps
- **Fine-tuning**: Last few layers are fine-tuned

### Decoder (RNN with Attention)
- **Type**: LSTM with attention mechanism
- **Attention**: Bahdanau-style attention
- **Embedding Size**: 512
- **Hidden Size**: 512
- **Dropout**: 0.5

### Attention Mechanism
The model uses soft attention to focus on different parts of the image when generating each word. This allows the model to:
- Look at relevant image regions for each word
- Generate more accurate and contextually appropriate captions
- Provide interpretable visualizations of what the model is "looking at"

## Configuration

Key parameters in `config.py`:

**Training Parameters:**
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 4e-4 (use 1e-4 for fine-tuning)
- `NUM_EPOCHS`: 50 (use 5-15 for fine-tuning)
- `GRAD_CLIP`: 5.0

**Early Stopping:**
- `EARLY_STOP_PATIENCE`: 5 epochs
- `MIN_DELTA`: 0.001

**Model Architecture:**
- `EMBED_SIZE`: 512
- `HIDDEN_SIZE`: 512
- `ATTENTION_DIM`: 512
- `ENCODER_DIM`: 2048

## Transfer Learning Strategies

### Strategy 1: Full Fine-tuning
Fine-tune both encoder and decoder. Best when you have a large dataset.
```bash
python finetune.py --pretrained_path model.pt --learning_rate 1e-4
```

### Strategy 2: Feature Extraction
Freeze encoder, train only decoder. Best for small datasets or similar domains.
```bash
python finetune.py --pretrained_path model.pt --freeze_encoder --learning_rate 1e-4
```

### Strategy 3: Gradual Unfreezing
Start with frozen encoder, then gradually unfreeze layers (modify `finetune.py`).

### Strategy 4: Different Learning Rates
Use different learning rates for encoder and decoder (requires modifying optimizer setup).

## Training Tips

1. **GPU Recommended**: Training on CPU will be very slow
2. **Batch Size**: Adjust based on your GPU memory
3. **Fine-tuning**: 
   - Use lower learning rate (1e-4 or 1e-5)
   - Consider freezing encoder for small datasets
   - Needs fewer epochs than training from scratch
4. **Data Augmentation**: Already included (random crop, flip)
5. **Pretrained Models**: Use pretrained models for faster convergence

## Common Use Cases

### Use Case 1: Quick Start with Pretrained Model
```python
# Load pretrained model and generate captions immediately
encoder, decoder, vocab = load_pretrained_model(pretrained_path, vocab_path, device)
# Use inference.py to generate captions
```

### Use Case 2: Fine-tune for Medical Images
```bash
# 1. Prepare your medical image dataset in COCO format
# 2. Fine-tune with frozen encoder
python finetune.py --pretrained_path coco_model.pt --freeze_encoder --epochs 10
```

### Use Case 3: Transfer to Different Language
```python
# 1. Build vocabulary for new language
# 2. Adapt decoder to new vocabulary size
decoder = adapt_vocab_size(decoder, new_vocab_size, device)
# 3. Fine-tune on new language dataset
```

### Use Case 4: Share Your Trained Model
```bash
# Export your model for others to use
python export_pretrained.py --checkpoint best_model.pt --model_name my_model
# Share the pretrained_models directory
```

## Troubleshooting

**Out of Memory Error**: Reduce `BATCH_SIZE` in `config.py`

**Slow Training**: Ensure CUDA is available and being used

**Poor Results**: Train for more epochs or adjust hyperparameters

**Vocabulary Mismatch**: Use `adapt_vocab_size()` when transferring to different vocabulary

**Loading Pretrained Errors**: Ensure vocabulary file matches the pretrained model

## Performance Benchmarks

Training from scratch on MS COCO 2014:
- **Training time**: ~20-30 hours on single GPU (V100)
- **Convergence**: 20-30 epochs typically sufficient

Fine-tuning pretrained model:
- **Training time**: ~2-5 hours on single GPU
- **Convergence**: 5-10 epochs typically sufficient
- **Speed-up**: 5-10x faster than training from scratch

## References

- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
- Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering
- MS COCO: Common Objects in Context

## License

MIT License
