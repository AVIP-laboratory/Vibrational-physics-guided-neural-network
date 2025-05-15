# Data description
The VPGNN framework is trained using experimentally measured impulse response data from 3D-printed soft auxetic metamaterials. The dataset consists of time-domain signals and associated physical metadata, allowing the network to learn both temporal patterns and their physical context.

The data is structured as follows:

**Training Dataset:**

 ●   **Experimental Impulse Response Data**: Raw time-displacement signals collected from impact hammer tests.
 Experimental Impulse Response Data is located in the `/Experimental data/` folder and provides access to information on all unit cell sizes of the re-entrant and honeycomb.

 ●   **Physical Metadata**: Includes unit cell length (H), reentrant angle (θ), strut thickness (t), sampling frequency (Fs), and signal duration (Ts)

 ●   **Data Source**:  In-house measurements from the AVIP Lab
