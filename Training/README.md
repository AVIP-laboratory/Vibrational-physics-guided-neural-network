import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Concatenate, Input, Conv1D, Flatten

# 데이터 로드 및 전처리
data = pd.read_csv(r'D:\도현\AVIP LAB\1. AUXETIC\실험data\최종데이터 for Origin\emd\hc15/hc15_10_time_displacement.csv', header=None)
data.columns = ['Time', 'Displacement']
measured_displacement = data['Displacement'].values.astype(np.float64)

# 샘플링 주파수 설정
sampling_frequency = 12800
meta_time = 0.4
unit_cell_thickness = 1.0  # 단위 셀 두께 t
unit_cell_height = 15.0  # 단위 셀 길이 H
reentrant_angle = 60.0  # 재진입 각도 θ

# 메타데이터 배열 생성 (샘플링 주파수 + 기하학적 정보)
metadata = np.array([sampling_frequency, meta_time, unit_cell_thickness, unit_cell_height, reentrant_angle])


# 입력 데이터 구성 (변위 데이터만 사용)
conv_input_data = measured_displacement.reshape(1, -1, 1)
print(conv_input_data.shape)
metadata_input_data = metadata.reshape(1, -1)
# 신경망 모델 정의
class FrequencyDampingModel(Model):
    def __init__(self):
        super(FrequencyDampingModel, self).__init__()

        # 1D Convolutional Branch (하나만 유지)
        self.c11 = Conv1D(16, 128, strides=4, activation='relu')
        self.c12 = Conv1D(32, 128, strides=4, activation='relu')
        self.c13 = Conv1D(64, 128, strides=4, activation='relu')
        self.c14 = Conv1D(1, 1, strides=1, activation='relu')

        # Dense Layers for Feature Extraction
        self.dense1 = Dense(1024, activation='tanh')
        self.dense2 = Dense(512, activation='tanh')

        # Output layers
        self.output_freq = Dense(1)  # 고유 진동수
        self.output_damp = Dense(1)  # 감쇠비

    def call(self, inputs):
        conv_input, metadata_input = inputs  # Conv1D 입력과 메타데이터 입력을 분리

        # Conv1D Feature Extraction
        c1 = self.c11(conv_input)
        c2 = self.c12(c1)
        c3 = self.c13(c2)
        c4 = self.c14(c3)

        conv_features = Concatenate(axis=1)([c4])
        conv_features = Flatten()(conv_features)

        x = Concatenate(axis=1)([conv_features, metadata_input])

        x = self.dense1(x)
        x = self.dense2(x)

        freq = self.output_freq(x)
        damp = self.output_damp(x)

        return freq, damp


# 모델 학습을 위한 Trainer 클래스
class ModelTrainer:
    def __init__(self):
        self.F = 0.00026919  # 충격량
        self.m = 0.0025  # 질량
        self.lr = 0.005
        self.opt = Adam(learning_rate=self.lr)
        self.model = FrequencyDampingModel()
        self.model.build(input_shape=[(None, conv_input_data.shape[1], 1), (None, metadata_input_data.shape[1])])

        self.model.summary()

    def save_weights(self, path):
        self.model.save_weights(path + 'model_weights.h5')

    def load_weights(self, path):
        self.model.load_weights(path + 'model_weights.h5')

    def predict(self, inputs):
        return self.model(inputs)

    def custom_loss(self, y_true, y_pred):
        measured_displacement = y_true
        predicted_freq, predicted_damp = y_pred

        time_data = np.arange(measured_displacement.shape[1]) / sampling_frequency
        omega_n = predicted_freq
        zeta = 0.001 * predicted_damp
        omega_d = omega_n * tf.sqrt(1 - tf.square(zeta))

        y = -(self.F * tf.exp(-zeta * omega_n * time_data)) / (self.m * omega_d) * tf.sin(omega_d * time_data)
        loss = tf.reduce_sum(tf.square(measured_displacement - y))
        return loss

    # def train(self, conv_input_data, y_true, max_num):
    def train(self, conv_input_data, metadata_input_data, y_true, max_num):

        conv_input_data = tf.convert_to_tensor(conv_input_data, dtype=tf.float32)
        metadata_input_data = tf.convert_to_tensor(metadata_input_data, dtype=tf.float32)
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

        train_loss_history = []
        mid_freq_history = []
        mid_damp_history = []


        for iter in range(int(max_num)):
            with tf.GradientTape() as tape:
                # y_pred = self.model(conv_input_data)
                y_pred = self.model([conv_input_data, metadata_input_data])

                loss = self.custom_loss(y_true, y_pred)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

            train_loss_history.append([iter, loss.numpy()])
            print(f'iter={iter}, loss={loss.numpy()}')

            freq, damp = y_pred
            print(f'Frequency at iter={iter}: {freq.numpy().flatten()[0]}')
            print(f'Damping at iter={iter}: {damp.numpy().flatten()[0]}')
            mid_freq, mid_damp = self.predict([conv_input_data, metadata_input_data])
            mid_freq_history.append([iter, mid_freq.numpy().flatten()[0]])
            mid_damp_history.append([iter, mid_damp.numpy().flatten()[0]])

        # 결과 저장
        results_df = pd.DataFrame({
            'Iteration': [i[0] for i in train_loss_history],
            'Loss': [i[1] for i in train_loss_history],
            'Frequency': [i[1] for i in mid_freq_history],
            'Damping': [i[1] for i in mid_damp_history]
        })

# 모델 학습 실행
def main():
    max_num = 15000
    trainer = ModelTrainer()
    trainer.train(conv_input_data, metadata_input_data, measured_displacement.reshape(1, -1), max_num)

    save_dir = r'D:\도현\AVIP LAB\1. AUXETIC\2. Hammer & VPGNN\[CODE] pinn_auxetic\Auxetic_1DCNN_Meta'
    trainer.save_weights(save_dir)

    final_freq, final_damp = trainer.predict([conv_input_data, metadata_input_data])

    time_data = np.arange(measured_displacement.shape[0]) / sampling_frequency
    # 최적화된 물성치로 도출된 이론적 응답 계산
    F = 0.00026919
    m = 0.0025
    omega_n = final_freq.numpy().flatten()[0]
    zeta = 0.001 * final_damp.numpy().flatten()[0]
    omega_d = omega_n * np.sqrt(1 - np.square(zeta))
    predicted_y = -(F * np.exp(-zeta * omega_n * time_data)) / (m * omega_d) * np.sin(omega_d * time_data)

    r2 = r2_score(measured_displacement, predicted_y)
    print(f"R² Score: {r2:.4f}")

    # 실험 데이터와 비교 그래프 출력
    plt.figure()
    plt.plot(time_data, measured_displacement, label='Measured Displacement', color='black')
    plt.plot(time_data, predicted_y, label='Predicted Displacement', linestyle='--', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.title('Measured vs Predicted Displacement')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
