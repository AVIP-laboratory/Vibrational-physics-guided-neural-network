# Vibrational-physics-guided-neural-network
This repository provides a physics-guided deep learning framework (VPGNN) for analyzing the dynamic properties of 3D-printed soft auxetic metamaterials. The model integrates physical constraints with experimental impulse response data to directly estimate natural frequencies and damping ratios, supporting various structural configurations and physical metadata for enhanced accuracy and generalization.

# Highlights
● **Physics-Guided Learning**: Mass-spring-damper inspired loss function for physically consistent results

● **Metadata-Enhanced Predictions**: Integrates physical metadata for improved accuracy and generalization

● **Efficient Data Utilization**: Combines empirical data and analytical models for data-efficient learning

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

**Training and Validation Dataset:
**
 ●   **Experimental Impulse Response Data**: Raw time-displacement signals collected from impact hammer tests

 ●   **Physical Metadata**: Includes unit cell length (H), reentrant angle (θ), strut thickness (t), sampling frequency (Fs), and signal duration (Ts)

 ●   **Data Source**: In-house curated dataset (AVIP Lab, Heilbronn University)

**Test Dataset:
**
 ●   **Re-entrant Structures**: Varying unit cell geometries (e.g., 10 mm, 12 mm, 15 mm)

 ●   **Honeycomb Structures**: Different strut lengths and angles to evaluate generalization capability

 ●   **Benchmark Structures**: Additional auxetic configurations for cross-validation and robustness testing

