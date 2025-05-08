import tensorflow as tf
from pathlib import Path

# Adjust this to your data folder
root_dir = Path(r'C:\Users\matt\iCloudDrive\Family\Education\Matt\KBS\Modules\2025 T1\TECH3300 - Machine Learning Applications\Assessment 2\Assessment 2 Data')

IMG_SIZE    = (100, 100)
BATCH_SIZE  = 32
SEED        = 42
AUTOTUNE    = tf.data.AUTOTUNE
VALIDATION_SPLIT = 0.2

# 1. Create training + validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir / "train",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="training"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir / "train",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="validation"
)

# 2. Define preprocessing: normalize to [0,1]
normalization = tf.keras.layers.Rescaling(1./255)

# 3. (Optional) Data augmentation for training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def preprocess(ds, augment=False):
    ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = preprocess(train_ds, augment=True)
val_ds   = preprocess(val_ds)

# 4. Define test dataset 
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir / "test",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)
test_ds = test_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(AUTOTUNE)

print("Datasets ready:")
print(f" • train: {tf.data.experimental.cardinality(train_ds).numpy()} batches")
print(f" • val:   {tf.data.experimental.cardinality(val_ds).numpy()} batches")
print(f" • test:  {tf.data.experimental.cardinality(test_ds).numpy()} batches")
