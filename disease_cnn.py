"""
disease_cnn.py

CNN for leaf disease classification (healthy vs diseased).
AIBUY3A Business Analysis 3.2 Project - Deep Learning component.

Trains on leaf_dataset/train and leaf_dataset/val (see
generate_leaf_dataset.py). Architecture is deliberately simple
(3 conv blocks) - appropriate for a diploma-level demonstration and
easy to explain in a presentation, while still being a genuine CNN
rather than a single dense layer.

To use a REAL dataset later (e.g. PlantVillage from Kaggle), just
replace the leaf_dataset/ folder with the same train/<class>/ and
val/<class>/ structure - no code changes needed.
"""

import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from keras import layers, models # type: ignore

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "leaf_dataset"


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=42,
    )
    class_names = train_ds.class_names  # ['diseased', 'healthy'] alphabetical
    print("Class names (0/1 label order):", class_names)

    # Normalize pixel values to [0,1]
    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # Light augmentation on training data only - helps generalization
    # given the training set isn't huge
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
    ])
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model():
    # Architecture history, for the project write-up's "solution techniques"
    # discussion:
    # v1: Flatten() straight into Dense(64) at 128x128 input produced a
    #     16384 -> 64 dense layer (1M+ params) fed by unnormalized
    #     activations. Loss spiked early, network collapsed to always
    #     predicting one class (50% accuracy, 0% recall on "diseased").
    # v2: Fixed the instability with BatchNormalization + swapped Flatten
    #     for GlobalAveragePooling2D. Training stabilized, but GAP averages
    #     activations across the whole spatial map - it can tell "there is
    #     some non-green color present" but not "there is a spot pattern
    #     localized in one region," which is exactly what distinguishes a
    #     few disease spots from none. Recall on diseased leaves stayed
    #     around 31%.
    # v3 (this version): kept BatchNormalization for stability, but used
    #     four conv+pool blocks to shrink the spatial map to 8x8 before
    #     Flatten (8*8*64=4096 -> Dense(64), ~260K params - large enough
    #     to retain spatial layout, small enough to stay stable).
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),  # 128 -> 64

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),  # 64 -> 32

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),  # 32 -> 16

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),  # 16 -> 8

        layers.Flatten(),  # 8*8*64 = 4096
        layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # binary: healthy vs diseased
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("cnn_training_history.png", dpi=150)
    print("Saved cnn_training_history.png")


def main():
    train_ds, val_ds, class_names = load_datasets()
    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop]
    )

    val_loss, val_acc, val_prec, val_recall = model.evaluate(val_ds)
    print(f"\nFinal validation - accuracy: {val_acc:.3f}, precision: {val_prec:.3f}, "
          f"recall: {val_recall:.3f}")
    print("\nNote: recall on 'diseased' matters most in practice - missing a "
          "diseased leaf (false negative) is costlier than a false alarm.")

    plot_history(history)

    model.save("leaf_disease_cnn.keras")
    print("Saved model -> leaf_disease_cnn.keras")

    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))


if __name__ == "__main__":
    main()
