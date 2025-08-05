# Inspiration:  https://github.com/gregtozzi/deep_learning_celnav
# I wanted to train a neural net / visual tansformer but have hardware constraints - my GPU is an NVidia RTX 3060 with 12GB VRAM.
# I like the ability of the above to detect lat/long of the observer to within some distance (we'll see through testing) - it could
# be somewhat useful in detection of fakes as well as to detect where input is coming from in a crowd-sourced online system.
# This would mean the network could automatically integrate new nodes potentially.

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.mixed_precision as mixed_precision
import numpy as np
import cv2
import os
from ktrain import Learner
from ktrain import autofit
from datetime import datetime
import yaml
from swin_transformer import SwinTransformer
from haversine import haversine

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Haversine loss function (from original implementation)
def haversine_loss(y_true, y_pred):
    # y_true and y_pred are normalized [0,1] for lat/long
    # Denormalize to actual lat/long values
    lat_min, lat_max = 36.0, 40.0  # Example bounds (adjust as needed)
    lon_min, lon_max = -78.0, -74.0
    y_true_lat = y_true[:, 0] * (lat_max - lat_min) + lat_min
    y_true_lon = y_true[:, 1] * (lon_max - lon_min) + lon_min
    y_pred_lat = y_pred[:, 0] * (lat_max - lat_min) + lat_min
    y_pred_lon = y_pred[:, 1] * (lon_max - lon_min) + lon_min

    # Haversine distance in nautical miles
    dist = haversine((y_true_lat, y_true_lon), (y_pred_lat, y_pred_lon), unit='nmi')
    return tf.reduce_mean(dist)

# Custom gradient accumulation layer
class GradientAccumulation:
    def __init__(self, accum_steps):
        self.accum_steps = accum_steps
        self.gradients = None
        self.step = 0

    def accumulate_gradients(self, model, loss, optimizer):
        if self.gradients is None:
            self.gradients = [tf.zeros_like(w) for w in model.trainable_weights]
        
        # Compute gradients for current batch
        batch_gradients = tf.gradients(loss, model.trainable_weights)
        self.gradients = [g + bg for g, bg in zip(self.gradients, batch_gradients)]
        self.step += 1

        if self.step >= self.accum_steps:
            # Apply accumulated gradients
            optimizer.apply_gradients(zip(self.gradients, model.trainable_weights))
            # Reset gradients and step
            self.gradients = None
            self.step = 0

# Swin-Small model with detection and regression heads
def build_swin_model():
    # Swin-Small backbone
    base_model = SwinTransformer('swin_small_224', include_top=False, pretrained=True)
    
    # Input layers
    image_input = keras.layers.Input(shape=(224, 224, 1), name='image_input')
    time_input = keras.layers.Input(shape=(1,), name='time_input')
    
    # Process image through Swin-Small
    x = base_model(image_input)
    
    # Detection head for star filter (bounding boxes)
    detection_head = keras.layers.Dense(100 * 4, activation='sigmoid', name='detection_head')(x)  # 100 boxes, 4 coords each
    detection_class = keras.layers.Dense(100, activation='sigmoid', name='detection_class')(x)  # 100 star/non-star probabilities
    
    # Regression head for lat/long
    x = keras.layers.Concatenate()([x, time_input])
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    regression_head = keras.layers.Dense(2, activation='sigmoid', name='regression_head')(x)  # Normalized lat/long
    
    model = keras.Model(inputs=[image_input, time_input], 
                        outputs=[detection_head, detection_class, regression_head])
    return model

# Load and preprocess data (adapted from original)
def load_data(data_dir):
    images = []
    times = []
    lat_longs = []
    bboxes = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            # Parse filename: <lat>+<long>+<YYYY>-<MM>-<DD>T<hh>:<mm>:<ss>.png
            parts = filename.split('+')
            lat, lon = float(parts[0]), float(parts[1])
            time_str = parts[2].split('.')[0]
            time = np.datetime64(time_str)
            
            # Load and preprocess image
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            
            # Normalize time and lat/long
            time_norm = (time.astype(np.int64) - np.datetime64('2020-05-25T22:00:00').astype(np.int64)) / (4 * 3600 * 1e9)  # 4-hour range
            lat_norm = (lat - 36.0) / (40.0 - 36.0)
            lon_norm = (lon - -78.0) / (-74.0 - -78.0)
            
            # Placeholder for star annotations (replace with actual Stellarium star data)
            # Assume 100 bounding boxes per image (adjust based on dataset)
            box = np.zeros((100, 4), dtype=np.float32)  # [x_min, y_min, x_max, y_max]
            label = np.zeros(100, dtype=np.float32)  # 1 for star, 0 for non-star
            # TODO: Generate actual star bounding boxes using Stellarium catalog
            
            images.append(img)
            times.append(time_norm)
            lat_longs.append([lat_norm, lon_norm])
            bboxes.append(box)
            labels.append(label)
    
    return (np.array(images), np.array(times), np.array(lat_longs), 
            np.array(bboxes), np.array(labels))

# Training loop with gradient accumulation
def train_model(data_dir, model_dir, accum_steps=4):
    # Load data
    images, times, lat_longs, bboxes, labels = load_data(data_dir)
    
    # Split data
    train_split = 0.9
    n_train = int(len(images) * train_split)
    train_data = (images[:n_train], times[:n_train], 
                  {'detection_head': bboxes[:n_train], 
                   'detection_class': labels[:n_train], 
                   'regression_head': lat_longs[:n_train]})
    val_data = (images[n_train:], times[n_train:], 
                {'detection_head': bboxes[n_train:], 
                 'detection_class': labels[n_train:], 
                 'regression_head': lat_longs[n_train:]})
    
    # Build model
    model = build_swin_model()
    
    # Compile with mixed precision
    optimizer = keras.optimizers.Adam(learning_rate=2e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(optimizer=optimizer,
                  loss={'detection_head': 'mse', 
                        'detection_class': 'binary_crossentropy', 
                        'regression_head': haversine_loss},
                  loss_weights={'detection_head': 1.0, 
                                'detection_class': 1.0, 
                                'regression_head': 1.0})
    
    # Gradient accumulation
    ga = GradientAccumulation(accum_steps)
    
    # Custom training loop
    epochs = 52
    batch_size = 8  # Small batch size to fit 12GB VRAM
    steps_per_epoch = len(train_data[0]) // batch_size
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_images = train_data[0][start_idx:end_idx]
            batch_times = train_data[1][start_idx:end_idx]
            batch_targets = {k: v[start_idx:end_idx] for k, v in train_data[2].items()}
            
            with tf.GradientTape() as tape:
                predictions = model([batch_images, batch_times], training=True)
                loss = model.compute_loss([batch_images, batch_times], batch_targets, predictions)
                scaled_loss = optimizer.get_scaled_loss(loss)
            
            ga.accumulate_gradients(model, scaled_loss, optimizer)
    
    # Save model
    model.save(os.path.join(model_dir, 'swin_model.h5'))

if __name__ == '__main__':
    data_dir = '/data/image_train_val'
    model_dir = '/data/models'
    train_model(data_dir, model_dir)
