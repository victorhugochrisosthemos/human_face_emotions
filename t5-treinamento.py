# ============================================================
# TREINAMENTO DA CNN — CARREGANDO DIRETAMENTE DO DATASET
# ------------------------------------------------------------
# Carrega imagens 48x48 grayscale
# Normaliza para 0–1
# Divide em treino/val/test
# Cria CNN e treina
# Gera precision/recall/F1
# ============================================================

import os
import csv
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout, Flatten,
    BatchNormalization
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 1. Carregar dataset
# ============================================================

dataset_path = "dataset_pre_processado"
image_size = (48, 48)

X = []
y = []

classes = sorted(os.listdir(dataset_path))
class_to_label = {classe: i for i, classe in enumerate(classes)}

print("Classes mapeadas:", class_to_label)

for classe in classes:
    caminho_classe = os.path.join(dataset_path, classe)

    if not os.path.isdir(caminho_classe):
        continue

    print(f"Carregando imagens da classe: {classe}")

    for arquivo in os.listdir(caminho_classe):
        caminho_img = os.path.join(caminho_classe, arquivo)

        try:
            img = Image.open(caminho_img).convert("L")   # já é grayscale
            img = img.resize(image_size)                 # já é 48x48

            img_array = np.array(img, dtype=np.float32) / 255.0  # NORMALIZA
            img_array = np.expand_dims(img_array, axis=-1)       # canal único

            X.append(img_array)
            y.append(class_to_label[classe])

        except Exception as e:
            print("Erro ao carregar:", caminho_img, e)

X = np.array(X)
y = np.array(y)

print("\nDataset carregado:")
print("X:", X.shape)
print("y:", y.shape)

# ============================================================
# 2. Train / Validation / Test Split
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit realizado:")
print("Treino:", X_train.shape, y_train.shape)
print("Validação:", X_val.shape, y_val.shape)
print("Teste:", X_test.shape, y_test.shape)

# ============================================================
# 3. CNN
# ============================================================

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                 input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(5, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nResumo da CNN:")
model.summary()

# ============================================================
# 4. Callback para logs em uma linha + CSV
# ============================================================

csv_file = "training_log.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

class CustomLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {logs['loss']:.4f}  // "
            f"Train Acc: {logs['accuracy']:.4f} // "
            f"Val Loss: {logs['val_loss']:.4f} // "
            f"Val Acc: {logs['val_accuracy']:.4f}"
        )

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                logs["loss"], logs["accuracy"],
                logs["val_loss"], logs["val_accuracy"]
            ])

# ============================================================
# 5. Treinar
# ============================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[CustomLogger()],
    verbose=0
)

# ============================================================
# 6. Avaliação Final
# ============================================================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\nRESULTADOS NO TESTE:")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ============================================================
# 7. Precision / Recall / F1
# ============================================================

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nRELATÓRIO COMPLETO (TESTE):")
print(classification_report(
    y_test,
    y_pred,
    target_names=list(classes)
))

print("\nTreinamento concluído. training_log.csv gerado.")
