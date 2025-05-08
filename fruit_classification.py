# fruit_classification.py (single-file pipeline + baseline model)

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Adjust this to your data folder
root_dir = Path(r'C:\Users\matt\iCloudDrive\Family\Education\Matt\KBS\Modules\2025 T1\TECH3300 - Machine Learning Applications\Assessment 2\Assessment 2 Data')

seed             = 42
autotune         = tf.data.AUTOTUNE

# === Hyperparameters ===

# Optimisation parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 20

# Regularisation parameters
dropout_rate = 0.5

# Data-Related parameters
validation_split = 0.2
img_size = (100, 100)



# 1. Create raw datasets to capture class names
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir / "train",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=seed,
    validation_split=validation_split,
    subset="training"
)
class_names = raw_train_ds.class_names

raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir / "train",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=seed,
    validation_split=validation_split,
    subset="validation"
)

# 2. Define preprocessing layers
normalization = tf.keras.layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def preprocess(ds, augment=False):
    ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=autotune)
    return ds.cache().prefetch(autotune)

# 3. Apply preprocessing
train_ds = preprocess(raw_train_ds, augment=True)
val_ds   = preprocess(raw_val_ds)

# 4. Define and preprocess test dataset
test_raw = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir / "test",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)
test_ds = test_raw.map(lambda x, y: (normalization(x), y)).cache().prefetch(autotune)

print("Datasets ready:")
print(f" • train: {tf.data.experimental.cardinality(train_ds).numpy()} batches")
print(f" • val:   {tf.data.experimental.cardinality(val_ds).numpy()} batches")
print(f" • test:  {tf.data.experimental.cardinality(test_ds).numpy()} batches")

# === Baseline CNN Model ===
num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.Input(shape=img_size + (3,)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

# Plot training curves
plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss"); plt.legend(); plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy"); plt.legend(); plt.show()

# Save the trained model
model_name = "baseline_fruit_classifier"
model.save(model_name, save_format="keras")
print(model_name)
