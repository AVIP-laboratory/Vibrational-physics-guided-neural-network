import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

file_list = [
]

labels = {
}


BATCH_SIZE = 4
EPOCHS     = 5000
LR         = 5e-3
W_FREQ     = 1e-1
W_DAMP     = 10
NORMALIZE_PER_SAMPLE = True

SAVE_DIR = r""
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "data driven.h5")


def load_disp(fp):
    df = pd.read_csv(fp, header=None)
    if df.shape[1] < 2:
    return df.iloc[:,1].astype(np.float64).values

series_list, y_list, used_files = [], [], []
for fp in file_list:
    fname = os.path.basename(fp)
    if fname not in labels:
    disp = load_disp(fp)
    series_list.append(disp)
    y_list.append([labels[fname]["omega_n"], labels[fname]["zeta"]])
    used_files.append(fname)

T = min(len(s) for s in series_list)
X = np.stack([s[:T] for s in series_list], axis=0)
y = np.array(y_list, dtype=np.float32)

if NORMALIZE_PER_SAMPLE:
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    X = (X - mu) / std

X = X[..., None].astype(np.float32)

test_files = [
  ]
test_idx = [used_files.index(f) for f in test_files]

X_test = X[test_idx]
y_test = y[test_idx]
f_test = [used_files[i] for i in test_idx]

train_idx = [i for i in range(len(used_files)) if i not in test_idx]
X_train_val = X[train_idx]
y_train_val = y[train_idx]
f_train_val = [used_files[i] for i in train_idx]

X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(
    X_train_val, y_train_val, f_train_val, test_size=3, random_state=42, shuffle=True
)

def make_ds(X, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds   = make_ds(X_val,   y_val,   shuffle=False)
test_ds  = make_ds(X_test,  y_test,  shuffle=False)

class FrequencyDampingModel(Model):
    def __init__(self):
        super().__init__()
        self.c11 = Conv1D(16, 128, strides=4, activation='relu')
        self.c12 = Conv1D(32, 128, strides=4, activation='relu')
        self.c13 = Conv1D(64, 128, strides=4, activation='relu')
        self.c14 = Conv1D(1,   1,   strides=1, activation='relu')
        self.dense1 = Dense(1024, activation='tanh')
        self.dense2 = Dense(512,  activation='tanh')
        self.out_freq = Dense(1)
        self.out_damp = Dense(1)

    def call(self, x, training=False):
        x = self.c11(x)
        x = self.c12(x)
        x = self.c13(x)
        x = self.c14(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        f = self.out_freq(x)
        z = self.out_damp(x)
        return f, z

model = FrequencyDampingModel()
model.build(input_shape=(None, T, 1))
model.summary()

opt = Adam(learning_rate=LR)

@tf.function
def train_step(x, y):
    y_f = y[:, :1]
    y_z = y[:, 1:2]
    with tf.GradientTape() as tape:
        p_f, p_z = model(x, training=True)
        lf = tf.reduce_mean(tf.square(y_f - p_f))
        lz = tf.reduce_mean(tf.square(y_z - p_z))
        loss = W_FREQ*lf + W_DAMP*lz
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(x, y):
    y_f = y[:, :1]
    y_z = y[:, 1:2]
    p_f, p_z = model(x, training=False)
    lf = tf.reduce_mean(tf.square(y_f - p_f))
    lz = tf.reduce_mean(tf.square(y_z - p_z))
    loss = W_FREQ*lf + W_DAMP*lz
    return loss

best_val = np.inf
train_hist, val_hist = [], []

for epoch in range(1, EPOCHS+1):
    train_losses = [float(train_step(xb, yb).numpy()) for xb, yb in train_ds]
    tl = np.mean(train_losses) if train_losses else np.nan

    val_losses = [float(val_step(xb, yb).numpy()) for xb, yb in val_ds]
    vl = np.mean(val_losses) if val_losses else np.nan

    train_hist.append(tl)
    val_hist.append(vl)

    if epoch % 50 == 0 or epoch <= 10:
    if not np.isnan(vl) and vl < best_val:
        best_val = vl
        model.save_weights(SAVE_PATH)


hist_df = pd.DataFrame({
    "epoch": np.arange(1, len(train_hist)+1),
    "train_loss": train_hist,
    "val_loss": val_hist
})

plt.figure()
plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (weighted MSE)")
plt.title("Training Curve (ω_n & ζ)")
plt.legend()
plt.grid(True)
plot_path = os.path.join(SAVE_DIR, "training_curve.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot:", plot_path)
