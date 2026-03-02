# Arabic Letter Classification

Data mining project for classifying Arabic letters using deep learning. The notebook compares multiple transfer learning models and implements a custom CNN and Vision Transformer.

## What's in the Notebook

The `data-mining-project.ipynb` file contains:

1. **Data Preprocessing**
   - Loads images from directories (BMP/JPG/PNG formats)
   - Resizes all images to 128x128 RGB
   - Filters classes with at least 100 images
   - One-hot encodes labels for 56 classes
   - Splits data 80/20 train/test with stratification

2. **Transfer Learning Models Tested**
   - VGG16
   - VGG19
   - InceptionV3
   - ResNet50
   - ResNet101
   - ResNet152
   - DenseNet121
   - DenseNet169
   - DenseNet201

   Each model uses ImageNet weights with frozen layers, adds GlobalAveragePooling2D, Dense(512) with L2 regularization, Dropout(0.3), and outputs to 56 classes.

3. **Custom CNN**
   - 3 Conv2D layers (32, 64, 128 filters)
   - MaxPooling2D and Dropout(0.25)
   - Flatten and Dense(128)
   - Output Dense(56) with softmax

4. **Data Augmentation**
   - ImageDataGenerator with 10-degree rotation
   - Tested with VGG16 and ResNet152

5. **Fine-Tuning**
   - VGG16 with all layers trainable

6. **Vision Transformer**
   - Uses `google/vit-base-patch16-224`
   - Resizes images to 224x224
   - Freezes encoder, trains classifier
   - Evaluates with accuracy, precision, recall, F1

## How to Use

### Prerequisites
- Python with numpy, pandas, matplotlib, PIL
- TensorFlow/Keras
- PyTorch and transformers (for ViT section)
- scikit-learn

### Running the Code

1. **Update the data path**: Change `/kaggle/input/` in the notebook to your local dataset directory

2. **Run preprocessing cells**: Execute cells to load and prepare data

3. **Choose a model section**: Run any of the model sections to train:
   ```python
   # Example from notebook
   base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128,3))
   trainning(MyModel(base_model))
   ```

4. **Optional data augmentation**:
   ```python
   data_gen = ImageDataGenerator(rotation_range=10)
   trainning(MyModel(base_model), data_gen)
   ```

### Training Details

- **Batch size**: 128
- **Epochs**: 10-26 depending on model
- **Callbacks**: EarlyStopping and ReduceLROnPlateau
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Loss**: Categorical crossentropy
- **Optimizer**: Adam

## File Structure

```
data-mining-project.ipynb    # Main notebook
README.md                    # This file
```

## Tech Stack

- TensorFlow/Keras (transfer learning and CNN)
- PyTorch + Transformers (Vision Transformer)
- NumPy, Pandas, Matplotlib
- PIL for image processing
- scikit-learn for splits and metrics

