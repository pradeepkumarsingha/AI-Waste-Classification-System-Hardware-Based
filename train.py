# import os
# import tensorflow as tf
# import matplotlib.pyplot as plt

# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # =========================
# # CONFIGURATION
# # =========================

# DATASET_DIR = "D:\AI Hardware\dataset-resized"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 15

# # =========================
# # LOAD DATASET
# # =========================

# train_dataset = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# validation_dataset = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# class_names = train_dataset.class_names

# print("Classes:", class_names)

# # =========================
# # PERFORMANCE
# # =========================

# AUTOTUNE = tf.data.AUTOTUNE

# train_dataset = train_dataset.prefetch(AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# # =========================
# # DATA AUGMENTATION
# # =========================

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomContrast(0.1)
# ])

# # =========================
# # PRETRAINED MODEL
# # =========================

# base_model = MobileNetV2(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(224, 224, 3)
# )

# # Freeze pretrained layers
# base_model.trainable = False

# # =========================
# # BUILD MODEL
# # =========================

# inputs = layers.Input(shape=(224, 224, 3))

# x = data_augmentation(inputs)

# x = preprocess_input(x)

# x = base_model(x, training=False)

# x = layers.GlobalAveragePooling2D()(x)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(
#     len(class_names),
#     activation="softmax"
# )(x)

# model = models.Model(inputs, outputs)

# # =========================
# # COMPILE
# # =========================

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # =========================
# # TRAIN
# # =========================

# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=EPOCHS
# )

# # =========================
# # SAVE MODEL
# # =========================

# os.makedirs("models", exist_ok=True)

# model.save("models/waste_classifier.keras")

# # Save class names
# with open("models/classes.txt", "w") as f:
#     for class_name in class_names:
#         f.write(class_name + "\n")

# print("\nModel saved successfully!")
# print("Classes:", class_names)

# # =========================
# # PLOT ACCURACY
# # =========================

# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Waste Classification Accuracy")

# plt.legend()
# plt.show()

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# CONFIGURATION
# =========================

DATASET_DIR = r"D:\AI Hardware\dataset-resized"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 20


# =========================
# LOAD DATASET
# =========================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print("Classes:", class_names)


# =========================
# PERFORMANCE
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.10)
])


# =========================
# PRETRAINED MODEL
# =========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the complete base model initially
base_model.trainable = False


# =========================
# BUILD MODEL
# =========================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = preprocess_input(x)

# Keep the base model in inference mode during the first stage
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)


# =========================
# CALLBACKS
# =========================

os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_waste_classifier.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# =========================
# STAGE 1:
# TRAIN CLASSIFICATION HEAD
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history_initial = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)


# =========================
# STAGE 2:
# FINE-TUNE LAST LAYERS
# =========================

base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


print("\nTrainable layers after fine-tuning setup:")
for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)


# Recompile after changing trainable layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history_finetune = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)


# =========================
# SAVE FINAL MODEL
# =========================

model.save("models/waste_classifier_finetuned.keras")

with open("models/classes.txt", "w") as f:
    for class_name in class_names:
        f.write(class_name + "\n")

print("\nFine-tuned model saved successfully!")
print("Classes:", class_names)


# =========================
# COMBINE TRAINING HISTORY
# =========================

accuracy = (
    history_initial.history["accuracy"]
    + history_finetune.history["accuracy"]
)

val_accuracy = (
    history_initial.history["val_accuracy"]
    + history_finetune.history["val_accuracy"]
)

loss = (
    history_initial.history["loss"]
    + history_finetune.history["loss"]
)

val_loss = (
    history_initial.history["val_loss"]
    + history_finetune.history["val_loss"]
)


# =========================
# PLOT RESULTS
# =========================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.axvline(
    x=INITIAL_EPOCHS - 1,
    color="red",
    linestyle="--",
    label="Fine-tuning starts"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.axvline(
    x=INITIAL_EPOCHS - 1,
    color="red",
    linestyle="--",
    label="Fine-tuning starts"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()