# src/train.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os, numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Paths
DATA_DIR = "data/processed"
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.experimental.AUTOTUNE
MODEL_DIR = "models/best_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Create datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", seed=42
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", seed=42
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", seed=42
)

# Prefetch
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.06),
    layers.RandomContrast(0.06),
])

# Build model
def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE,3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)   # EfficientNet preprocess
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x, pooling="avg")
    base.trainable = False
    x = layers.Dropout(0.4)(base.output)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
)

# Compute class weights (important for imbalance)
# Extract labels from train_ds
y_train = np.concatenate([y.numpy() for x,y in train_ds], axis=0)
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(k): float(v) for k,v in zip(classes, cw)}
print("Class weights:", class_weights)

# Callbacks
cb = [
    callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best.h5"), save_best_only=True, monitor="val_auc", mode="max"),
    callbacks.EarlyStopping(monitor="val_auc", patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, verbose=1)
]

# Train head
history = model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=class_weights, callbacks=cb)

# Fine-tune: unfreeze some layers
base_model = model.layers[2]  # EfficientNetB0 base is inserted as layer 2 due to our setup
base_model.trainable = True
# Freeze first N layers if you want
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weights, callbacks=cb)

# Save final model (SavedModel format)
model.save(MODEL_DIR)
print("Saved model to", MODEL_DIR)
