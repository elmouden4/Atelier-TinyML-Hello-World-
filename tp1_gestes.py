import os
import sys
import time
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# 1.1.1 Initialisation et reproductibilité
# ==============================================================================
APOGEE = 25034860  # <-- REMPLACER PAR TON NUMÉRO D'APOGÉE
assert isinstance(APOGEE, int) and 1 < APOGEE <= 99_999_999

SEED = APOGEE
FS, N, G0 = 50, 64, 9.80665
CLASS_NAMES = ['Repos', 'Cercle', 'Ligne', 'Zigzag']

RNG = np.random.default_rng(SEED)
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

Path("artifacts").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

print("Executable :", sys.executable)
print("Python     :", sys.version.split()[0])
print("Versions   : TF =", tf.__version__, "| NumPy =", np.__version__)

# ==============================================================================
# 1.1.2 Fonctions fournies et génération des gestes
# ==============================================================================
def apply_orientation(data, rng):
    """Orientation fixe pendant une fenetre, sortie de meme forme en g."""
    theta = rng.uniform(0, 2 * np.pi)
    tilt = rng.uniform(-0.12, 0.12)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt),  np.cos(tilt)]
    ])
    return np.asarray(data, dtype=np.float64) @ (Rz @ Rx).T


def apply_gain_bias(data, rng):
    """Un gain et un biais par axe, constants pendant la fenetre."""
    gain = rng.uniform(0.95, 1.05, (1, 3))
    bias = rng.uniform(-0.08, 0.08, (1, 3))
    return data * gain + bias


def add_noise(data, t, rng, stress=False):
    """Bruit gaussien; stress ajoute plus de bruit et une derive lente."""
    sigma = rng.uniform(0.003, 0.025)
    if stress:
        sigma *= 1.7
    result = data + rng.normal(0, sigma, data.shape)
    if stress:
        phi_s = rng.uniform(0, 2 * np.pi)
        drift = 0.025 * np.sin(2 * np.pi * 0.25 * t + phi_s)
        result += drift[:, None]
    return result.astype(np.float32)


def generate_gesture(gesture_type, rng=None, num_samples=64, sample_rate=50, stress=False):
    """Retourne une fenetre brute en g (num_samples, 3) et son label."""
    if rng is None:
        rng = RNG

    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    x = np.zeros(num_samples, dtype=np.float64)
    y = np.zeros(num_samples, dtype=np.float64)
    z = np.ones(num_samples, dtype=np.float64)
    phi = rng.uniform(0, 2 * np.pi)

    if gesture_type == 0:
        pass
    elif gesture_type == 1:
        f = rng.uniform(0.8, 1.6)
        r = rng.uniform(0.07, 0.14)
        s = rng.choice([-1, 1])
        eta = rng.uniform(0.8, 1.2)
        A = (r * (2 * np.pi * f) ** 2) / G0
        x = A * np.cos(2 * np.pi * f * t + phi)
        y = s * eta * A * np.sin(2 * np.pi * f * t + phi)
    elif gesture_type == 2:
        L = rng.uniform(0.12, 0.28)
        D = rng.uniform(0.55, 0.85)
        t0 = rng.uniform(0.05, 0.30)
        A = (2 * np.pi * L) / (G0 * (D ** 2))
        u = (t - t0) / D
        mask = (u >= 0.0) & (u <= 1.0)
        x[mask] = A * np.sin(2 * np.pi * u[mask])
    elif gesture_type == 3:
        f = rng.uniform(2.5, 4.0)
        d = rng.uniform(0.008, 0.025)
        A = (d * (2 * np.pi * f) ** 2) / G0
        x = -A * np.sin(2 * np.pi * f * t + phi)
    else:
        raise ValueError("Type de geste invalide")

    data = np.column_stack((x, y, z))
    data = apply_orientation(data, rng)
    data = apply_gain_bias(data, rng)
    data = add_noise(data, t, rng, stress=stress)
    return data, int(gesture_type)


# ==============================================================================
# 1.1.3 & 1.1.4 Prétraitement et création des datasets
# ==============================================================================
def preprocess_window(raw_g):
    """Centre chaque axe temporellement, ecrete dans [-4, 4] g, retourne float32."""
    arr = np.asarray(raw_g, dtype=np.float32)
    assert arr.shape == (N, 3) and np.all(np.isfinite(arr))
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    return np.clip(centered, -4.0, 4.0).astype(np.float32)


def create_dataset(samples_per_class=300, seed=SEED, stress=False):
    """Genere un dataset equilibre (brut, pretraite, labels) melange conjointement."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for gesture_type in range(len(CLASS_NAMES)):
        for _ in range(samples_per_class):
            data, label = generate_gesture(
                gesture_type, rng=rng, num_samples=N, sample_rate=FS, stress=stress
            )
            X_list.append(data)
            y_list.append(label)

    X_raw = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    X_preprocessed = np.array([preprocess_window(w) for w in X_raw], dtype=np.float32)

    perm = rng.permutation(len(y))
    return X_raw[perm], X_preprocessed[perm], y[perm]


# Diagnostic + Train / Val / Test / Stress
X_diag_raw, X_diag, y_diag = create_dataset(samples_per_class=300, seed=SEED, stress=False)
X_train_raw, X_train, y_train = create_dataset(samples_per_class=600, seed=SEED + 1, stress=False)
X_val_raw,   X_val,   y_val   = create_dataset(samples_per_class=150, seed=SEED + 2, stress=False)
X_test_raw,  X_test,  y_test  = create_dataset(samples_per_class=150, seed=SEED + 3, stress=False)
X_stress_raw, X_stress, y_stress = create_dataset(samples_per_class=150, seed=SEED + 4, stress=True)

# ==============================================================================
# 1.2.1 Architecture du modèle CNN 1D
# ==============================================================================
def create_tinyml_model(input_shape=(64, 3), num_classes=4):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', activation='relu'),
        layers.Conv1D(filters=16, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.10),
        layers.Dense(num_classes, activation='softmax')
    ], name="TinyML_Gesture_CNN")
    return model


model = create_tinyml_model(input_shape=(N, 3), num_classes=len(CLASS_NAMES))
model.summary()
assert model.count_params() < 1000, f"Trop de parametres: {model.count_params()}"

# ==============================================================================
# 1.2.2 Compilation et entraînement
# ==============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nKeras Test Accuracy: {test_accuracy * 100:.2f}%")

# Courbes Loss / Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Training and Validation Loss')

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)
axes[1].set_title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig("figures/courbes_apprentissage.png", dpi=150)
plt.close(fig)

# ==============================================================================
# 1.2.3 Matrice de confusion et rapport de classification
# ==============================================================================
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes, labels=range(4))
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv("results/confusion_matrix_keras_test.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Prediction')
plt.ylabel('Verite terrain')
plt.title('Matrice de confusion - Modele Keras (Test)')
plt.tight_layout()
plt.savefig("figures/matrice_confusion_test.png", dpi=150)
plt.close()

print("\n=== Rapport de classification (Keras Test) ===")
print(classification_report(y_test, y_pred_classes, labels=range(4),
                            zero_division=0, target_names=CLASS_NAMES))

# ==============================================================================
# 1.3.1 Sauvegarde Float32 (Keras .h5 et TFLite Float32)
# ==============================================================================
output_dir = Path('artifacts')
model_filename = output_dir / f"gesture_model_float32_{APOGEE}.h5"
model.save(model_filename)
float32_h5_size = os.path.getsize(model_filename)
print(f"Modele Keras sauvegarde : {model_filename} ({float32_h5_size / 1024:.2f} KiB)")

converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter_float.convert()
float_tflite_path = output_dir / "gesture_float.tflite"
float_tflite_path.write_bytes(tflite_float_model)
float32_tflite_size = os.path.getsize(float_tflite_path)
print(f"Modele TFLite Float32   : {float_tflite_path} ({float32_tflite_size / 1024:.2f} KiB)")

# ==============================================================================
# 1.3.2 Quantification INT8 avec PTQ (100 exemples calibres : 25/classe)
# ==============================================================================
def representative_dataset():
    rng_cal = np.random.default_rng(SEED + 99)
    cal_indices = []
    for cls_idx in range(4):
        cls_idxs = np.where(y_train == cls_idx)[0]
        chosen = rng_cal.choice(cls_idxs, size=25, replace=False)
        cal_indices.extend(chosen.tolist())
    rng_cal.shuffle(cal_indices)

    for idx in cal_indices:
        sample = X_train[idx][None, :, :].astype(np.float32)  # (1, 64, 3)
        yield [sample]


converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8_model = converter_int8.convert()
int8_tflite_path = output_dir / "gesture_int8.tflite"
int8_tflite_path.write_bytes(tflite_int8_model)
int8_tflite_size = os.path.getsize(int8_tflite_path)
print(f"Modele TFLite INT8      : {int8_tflite_path} ({int8_tflite_size / 1024:.2f} KiB)")
assert int8_tflite_size < 10 * 1024, f"Taille INT8 > 10 KiB ({int8_tflite_size} octets)"

print("\n=== Inspection du graphe TFLite INT8 ===")
tf.lite.experimental.Analyzer.analyze(model_path=str(int8_tflite_path))

# ==============================================================================
# 1.3.3 Validation des modèles TFLite Float32 et INT8
# ==============================================================================
def evaluate_tflite_float(tflite_model_path, X, y):
    it = tf.lite.Interpreter(
        model_path=str(tflite_model_path), num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
    )
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    assert di['dtype'] == np.float32 and do['dtype'] == np.float32

    pred, scores = [], []
    for win in X:
        it.set_tensor(di['index'], win[None].astype(np.float32))
        it.invoke()
        p = it.get_tensor(do['index'])[0].astype(np.float32)
        pred.append(int(np.argmax(p)))
        scores.append(p)
    pred, scores = np.array(pred), np.array(scores)
    return float(np.mean(pred == y)), pred, scores, 0.0


def evaluate_tflite_int8(tflite_model_path, X, y):
    it = tf.lite.Interpreter(
        model_path=str(tflite_model_path), num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
    )
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    print("Input details INT8 :", di['shape'], di['dtype'], di['quantization'])
    print("Output details INT8:", do['shape'], do['dtype'], do['quantization'])

    assert di['dtype'] == np.int8 and do['dtype'] == np.int8
    assert di['shape'].tolist() == [1, 64, 3] and do['shape'].tolist() == [1, 4]
    scale, zero = di['quantization']
    out_scale, out_zero = do['quantization']
    assert np.isfinite(scale) and scale > 0 and np.isfinite(out_scale) and out_scale > 0

    pred, scores, saturated = [], [], 0
    for win in X:
        u = win.astype(np.float32) / scale + zero
        saturated += int(np.count_nonzero((u < -128) | (u > 127)))
        rounded = np.sign(u) * np.floor(np.abs(u) + 0.5)
        q = np.clip(rounded, -128, 127).astype(np.int8)
        it.set_tensor(di['index'], q[None])
        it.invoke()
        p = (it.get_tensor(do['index'])[0].astype(np.float32) - out_zero) * out_scale
        pred.append(int(np.argmax(p)))
        scores.append(p)
    pred, scores = np.array(pred), np.array(scores)
    return float(np.mean(pred == y)), pred, scores, float(saturated / X.size)


def compute_threshold_metrics(y_true, y_pred, scores, threshold=0.60):
    max_scores = np.max(scores, axis=1)
    accepted = max_scores >= threshold
    coverage = float(np.mean(accepted))
    acc_accepted = float(np.mean(y_pred[accepted] == y_true[accepted])) if np.any(accepted) else np.nan
    return coverage, acc_accepted


eval_rows = []
for dataset_name, X_eval, y_eval in [("Test", X_test, y_test), ("Stress", X_stress, y_stress)]:
    acc_f, pred_f, scores_f, sat_f = evaluate_tflite_float(float_tflite_path, X_eval, y_eval)
    acc_i, pred_i, scores_i, sat_i = evaluate_tflite_int8(int8_tflite_path, X_eval, y_eval)

    recalls_f = recall_score(y_eval, pred_f, average=None, labels=[0, 1, 2, 3])
    recalls_i = recall_score(y_eval, pred_i, average=None, labels=[0, 1, 2, 3])
    cov_f, acc_acc_f = compute_threshold_metrics(y_eval, pred_f, scores_f, threshold=0.60)
    cov_i, acc_acc_i = compute_threshold_metrics(y_eval, pred_i, scores_i, threshold=0.60)

    eval_rows.append({
        "Dataset": dataset_name,
        "Modele": "Float32",
        "Accuracy (%)": round(acc_f * 100, 2),
        "Baisse (pts)": 0.0,
        "Rappel_Repos (%)": round(recalls_f[0] * 100, 2),
        "Rappel_Cercle (%)": round(recalls_f[1] * 100, 2),
        "Rappel_Ligne (%)": round(recalls_f[2] * 100, 2),
        "Rappel_Zigzag (%)": round(recalls_f[3] * 100, 2),
        "Saturation (%)": 0.0,
        "Couverture_0.60 (%)": round(cov_f * 100, 2),
        "Acc_Acceptee_0.60 (%)": round(acc_acc_f * 100, 2)
    })
    eval_rows.append({
        "Dataset": dataset_name,
        "Modele": "INT8",
        "Accuracy (%)": round(acc_i * 100, 2),
        "Baisse (pts)": round((acc_f - acc_i) * 100, 2),
        "Rappel_Repos (%)": round(recalls_i[0] * 100, 2),
        "Rappel_Cercle (%)": round(recalls_i[1] * 100, 2),
        "Rappel_Ligne (%)": round(recalls_i[2] * 100, 2),
        "Rappel_Zigzag (%)": round(recalls_i[3] * 100, 2),
        "Saturation (%)": round(sat_i * 100, 4),
        "Couverture_0.60 (%)": round(cov_i * 100, 2),
        "Acc_Acceptee_0.60 (%)": round(acc_acc_i * 100, 2)
    })

df_eval = pd.DataFrame(eval_rows)
df_eval.to_csv("results/evaluation_float_vs_int8.csv", index=False)
print("\n=== Tableau comparatif Float32 vs INT8 ===")
print(df_eval.to_string(index=False))

# ==============================================================================
# 1.3.4 Analyse des performances (Latence PC)
# ==============================================================================
def benchmark_tflite(tflite_model_path, X, warmup=20, repeats=200):
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_model_path), num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    assert len(X) > 0 and warmup >= 0 and repeats >= 100

    inputs = []
    for window in X:
        data = window[None].astype(np.float32)
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            assert scale > 0
            u = data / scale + zero_point
            data = np.clip(np.sign(u) * np.floor(np.abs(u) + 0.5), -128, 127).astype(np.int8)
        inputs.append(data)

    durations = []
    for i in range(warmup + repeats):
        interpreter.set_tensor(input_details['index'], inputs[i % len(inputs)])
        start = time.perf_counter_ns()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if i >= warmup:
            durations.append(elapsed_ms)
    return float(np.median(durations)), float(np.percentile(durations, 95))


lat_f, p95_f = benchmark_tflite(float_tflite_path, X_test, warmup=20, repeats=200)
lat_i, p95_i = benchmark_tflite(int8_tflite_path, X_test, warmup=20, repeats=200)

df_perf = pd.DataFrame([
    {"Modele": "Float32", "Taille (KiB)": round(float32_tflite_size / 1024, 2),
     "Mediane_PC (ms)": round(lat_f, 4), "p95_PC (ms)": round(p95_f, 4)},
    {"Modele": "INT8", "Taille (KiB)": round(int8_tflite_size / 1024, 2),
     "Mediane_PC (ms)": round(lat_i, 4), "p95_PC (ms)": round(p95_i, 4)}
])
df_perf.to_csv("results/benchmark_latence_pc.csv", index=False)

print("\n=== Benchmark Latence PC (BUILTIN_REF, 1 thread) ===")
print("CPU / OS :", platform.processor(), "|", platform.platform())
print(df_perf.to_string(index=False))
print(f"Ratio des medianes (Float32 / INT8) : {lat_f / lat_i:.2f}x")
print(f"Ratio des tailles  (Float32 / INT8) : {float32_tflite_size / int8_tflite_size:.2f}x")

# Sauvegarde des metadonnees pour le contrat TP2
it_check = tf.lite.Interpreter(model_path=str(int8_tflite_path))
it_check.allocate_tensors()
in_det = it_check.get_input_details()[0]
out_det = it_check.get_output_details()[0]

contract_info = {
    "apogee": APOGEE,
    "seed": SEED,
    "python_version": sys.version.split()[0],
    "tensorflow_version": tf.__version__,
    "numpy_version": np.__version__,
    "input_shape": [1, 64, 3],
    "sampling_rate_hz": FS,
    "classes": CLASS_NAMES,
    "total_params": int(model.count_params()),
    "h5_size_bytes": float32_h5_size,
    "float_tflite_size_bytes": float32_tflite_size,
    "int8_tflite_size_bytes": int8_tflite_size,
    "input_quant_scale": float(in_det['quantization'][0]),
    "input_quant_zero_point": int(in_det['quantization'][1]),
    "output_quant_scale": float(out_det['quantization'][0]),
    "output_quant_zero_point": int(out_det['quantization'][1]),
    "threshold_selected": 0.60
}
with open("results/contract_tp2.json", "w", encoding="utf-8") as f:
    json.dump(contract_info, f, indent=2)
print("\nFichiers generes avec succes dans artifacts/, figures/ et results/ !")