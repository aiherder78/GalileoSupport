import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.mixed_precision as mixed_precision
import numpy as np
import cv2
import os
from transformers import TFDetrForObjectDetection
from haversine import haversine

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Haversine loss (same as above)
def haversine_loss(y_true, y_pred):
    lat_min, lat_max = 36.0, 40.0
    lon_min, lon_max = -78.0, -74.0
    y_true_lat = y_true[:, 0] * (lat_max - lat_min) + lat_min
    y_true_lon = y_true[:, 1] * (lon_max - lon_min) + lon_min
    y_pred_lat = y_pred[:, 0] * (lat_max - lat_min) + lat_min
    y_pred_lon = y_pred[:, 1] * (lon_max - lon_min) + lon_min
    dist = haversine((y_true_lat, y_true_lon), (y_pred_lat, y_pred_lon), unit='nmi')
    return tf.reduce_mean(dist)

# Gradient accumulation (same as above)
class GradientAccumulation:
    def __init__(self, accum_steps):
        self.accum_steps = accum_steps
        self.gradients = None
        self.step = 0

    def accumulate_gradients(self, model, loss, optimizer):
        if self.gradients is None:
            self.gradients = [tf.zeros_like(w) for w in model.trainable_weights]
        batch_gradients = tf.gradients(loss, model.trainable_weights)
        self.gradients = [g + bg for g, bg in zip(self.gradients, batch_gradients)]
        self.step += 1
        if self.step >= self.accum_steps:
            optimizer.apply_gradients(zip(self.gradients, model.trainable_weights))
            self.gradients = None
            self.step = 0

# DETR model with regression head
def build_detr_model():
    # DETR backbone
    base_model = TFDetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
    
    # Input layers
    image_input = keras.layers.Input(shape=(224, 224, 1), name='image_input')
    time_input = keras.layers.Input(shape=(1,), name='time_input')
    
    # Convert grayscale to RGB for DETR (expects 3 channels)
    x = keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(image_input)
    outputs = base_model(x)
    
    # Detection outputs
    detection_head = outputs.logits  # Bounding boxes
    detection_class = outputs.pred_boxes  # Class probabilities
    
    # Regression head
    x = outputs.last_hidden_state[:, 0, :]  # Use CLS token or similar
    x = keras.layers.Concatenate()([x, time_input])
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    regression_head = keras.layers.Dense(2, activation='sigmoid', name='regression_head')(x)
    
    model = keras.Model(inputs=[image_input, time_input], 
                        outputs=[detection_head, detection_class, regression_head])
    return model

# Load data (same as above)
def load_data(data_dir):
    images, times, lat_longs, bboxes, labels = [], [], [], [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            parts = filename.split('+')
            lat, lon = float(parts[0]), float(parts[1])
            time_str = parts[2].split('.')[0]
            time = np.datetime64(time_str)
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            time_norm = (time.astype(np.int64) - np.datetime64('2020-05-25T22:00:00').astype(np.int64)) / (4 * 3600 * 1e9)
            lat_norm = (lat - 36.0) / (40.0 - 36.0)
            lon_norm = (lon - -78.0) / (-74.0 - -78.0)
            box = np.zeros((100, 4), dtype=np.float32)
            label = np.zeros(100, dtype=np.float32)
            images.append(img)
            times.append(time_norm)
            lat_longs.append([lat_norm, lon_norm])
            bboxes.append(box)
            labels.append(label)
    return (np.array(images), np.array(times), np.array(lat_longs), 
            np.array(bboxes), np.array(labels))

# Training loop (same as above)
def train_model(data_dir, model_dir, accum_steps=4):
    images, times, lat_longs, bboxes, labels = load_data(data_dir)
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
    model = build_detr_model()
    optimizer = keras.optimizers.Adam(learning_rate=2e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer,
                  loss={'detection_head': 'mse', 
                        'detection_class': 'binary_crossentropy', 
                        'regression_head': haversine_loss},
                  loss_weights={'detection_head': 1.0, 
                                'detection_class': 1.0, 
                                'regression_head': 1.0})
    ga = GradientAccumulation(accum_steps)
    epochs = 52
    batch_size = 8
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
    model.save(os.path.join(model_dir, 'detr_model.h5'))

if __name__ == '__main__':
    data_dir = '/data/image_train_val'
    model_dir = '/data/models'
    train_model(data_dir, model_dir)
