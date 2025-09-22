# Vibrational-physics-guided-neural-network
This repository provides a physics-guided deep learning framework (VPGNN) for analyzing the dynamic properties of 3D-printed soft auxetic metamaterials. The model integrates physical constraints with experimental impulse response data to directly estimate natural frequencies and damping ratios, supporting various structural configurations and physical metadata for enhanced accuracy and generalization.

# Highlights
● **Physics-Guided Learning**: Mass-spring-damper inspired loss function for physically consistent results

● **Metadata-Enhanced Predictions**: Integrates physical metadata for improved accuracy and generalization

● **Efficient Data Utilization**: Combines experimental data and analytical models for data-efficient learning

● **Noise-Robust Performance**: Demonstrated stable predictions under Gaussian noise (SNR = 0, -5, -10 dB)

● **Scalable and Interpretable**: Suitable for complex, nonlinear systems like soft auxetic metamaterials

● Result saving and visualization (CSV, Matplotlib)

# Requirements
● Python 3.9

● tenseorflow

● numpy

● matplotlib

# Data description
The VPGNN framework is trained using experimentally measured impulse response data from 3D-printed soft auxetic metamaterials. The dataset consists of time-domain signals and associated physical metadata, allowing the network to learn both temporal patterns and their physical context.

The data is structured as follows:

**Training Dataset:**

 ●   **Experimental Impulse Response Data**: Raw time-displacement signals collected from impact hammer tests.
 Experimental Impulse Response Data is located in the `/Experimental data/` folder and provides access to information on all unit cell sizes of the re-entrant and honeycomb.

 ●   **Physical Metadata**: Includes unit cell length (H), reentrant angle (θ), strut thickness (t), sampling frequency (Fs), and signal duration (Ts)

 ●   **Noisy Data**:  Impulse response signals with added Gaussian noise (SNR = 0, -5, -10 dB) were also included for robustness evaluation

 ●   **Data Source**:  In-house measurements from the AVIP Lab

 # Code description
### Training VPGNN
 `/Training/VPGNN_auxetic.py` is a code for deriving dynamic properties of 3D-printed soft auxetic metastructures. It contains information about the model architecture, loss function, and training process.

 # Saved model
 The `/trained model/` folder contains the trained models that constitute the VPGNN. The trained models are saved as `.h5` file.

# Training reults
![image](https://github.com/user-attachments/assets/1fe6ad42-a930-4290-b986-c6ccddeda4af)
Figure 1. Representative results of the VPGNN for the re-entrant auxetic specimen with a unit cell size of 10 mm. (a) Predicted dynamic properties according to training iterations: (a) natural frequency 𝜔<sub>𝑛</sub>, (b) damping ratio 𝜁. (c) Comparison between the measured impulse response and the reconstructed response using the predicted dynamic properties at different iterations.

# Noise injection test results
<img width="1128" height="911" alt="image" src="https://github.com/user-attachments/assets/1b98a637-0f55-4a89-ba13-cc262e3d6872" />
Figure 2. Comparison of measured and reconstructed impulse responses for the re-entrant auxetic specimen with a unit cell size of 10 mm under various SNR conditions using (a) VPGNN, (b) data-driven approach, (c) curve fitting, and (d) Prony’s method. 



