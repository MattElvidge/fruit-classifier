# Imports
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Adjust this to your data folder
root_dir = Path(r'C:\Users\matt\iCloudDrive\Family\Education\Matt\KBS\Modules\2025 T1\TECH3300 - Machine Learning Applications\Assessment 2\Assessment 2 Data')

# Variables
seed = 42 # For reproducibility

# === Hyperparameters ===

# Optimisation parameters
learning_rate = 1e-3 # Learning rate for Adam optimiser
batch_size = 32 # Batch size for training and validation datasets
num_epochs = 20 # Number of epochs for training

# Regularisation parameters
dropout_rate = 0.5 # Dropout rate for the fully connected layer

# Data-Related parameters
validation_split = 0.2 # Fraction of training data to reserve for validation
img_size = (100, 100) # Image size for resising (height, width)

# Preprocess function
def preprocess(ds, augment=False):
    ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)

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
test_ds = test_raw.map(lambda x, y: (normalization(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# 5. Print dataset information
print("Datasets ready:")
print(f" • train: {tf.data.experimental.cardinality(train_ds).numpy()} batches")
print(f" • val:   {tf.data.experimental.cardinality(val_ds).numpy()} batches")
print(f" • test:  {tf.data.experimental.cardinality(test_ds).numpy()} batches")

# === Baseline CNN Model ===

# 1. Define the model
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

# 2. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Train the model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]
# 4. Save the model (after training)
history = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

# 5. Plot model and evaluate the model on the test dataset
plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss"); plt.legend(); plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy"); plt.legend(); plt.show()

model.save("baseline_fruit_classifier.keras")
print(model)

# === Evaluate Performance ===

# 6. Evaluate on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 7. Confusion matrix
# Gather true labels and predictions
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

# Plot it
cm = confusion_matrix(y_true_labels, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(xticks_rotation="vertical")
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.show()
