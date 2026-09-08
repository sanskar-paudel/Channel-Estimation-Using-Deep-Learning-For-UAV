# Channel Estimation Using Deep Learning For UAV

This repository contains the core implementation, frameworks, and deep learning architectures for **pilot-free channel estimation** tailored for **UAV CP-OFDM** and **MIMO-OFDM** communication systems. 

This codebase is a direct accompaniment to our current research paper. It provides a modular framework where alternative wireless communication environments can be seamlessly integrated and evaluated.

>  **Note:** This repository is actively maintained as part of an ongoing research project. Following the formal completion and publication of the research paper, comprehensive baselines and alternative supervised Deep Neural Network (DNN) methods will be fully uploaded here.

---

##  Key Features

* **Hybrid Implementation:** Combines high-performance deep learning models in **Python (PyTorch/TensorFlow)** with dedicated verification/simulation engines in **MATLAB**.
* **Supervised & Unsupervised Learning:** Ready-to-use models covering both Supervised Autoencoders (AE) and Unsupervised Autoencoders equipped with advanced data-smoothening techniques.
* **UAV & MIMO Optimization:** Tailored specifically for modern high-mobility UAV environments utilizing Cyclic Prefix OFDM (CP-OFDM) and multi-antenna configurations.
* **Extensible Baseline Framework:** Designed cleanly as a simple base architecture to allow researchers to easily modify or plug in complex, custom channel matrices ($H$).

---

##  Codebase Structure

The repository currently hosts the core deep learning engines and modified scripts:

*  `Auto encoder (Supervised).py` — Python script implementing the fully supervised Autoencoder backbone for channel profile matching.
*  `Unsupervised AE with Smoothening.py` — Python script featuring our custom unsupervised Autoencoder pipeline integrated with post-processing smoothening for pilot-free estimation.
*  `Modified AE code.m` — Complementary MATLAB script utilized for channel generation, performance validation, or dataset alignment.

---

##  Simulation & Usage

### Python (Deep Learning Engine)
Ensure your environment satisfies standard data science dependencies (`numpy`, `scipy`, and your DL framework package). Run either the supervised or unsupervised tracking models directly:
```bash
python "Unsupervised AE with Smoothening.py"
```

### MATLAB (System Environment)
For running or adjusting the physical communication layer baseline configurations:
1. Open MATLAB and point your directory to this repository.
2. Initialize or interface your system matrices using:
   ```matlab
   run('Modified AE code.m')
   ```

---

##  Future Releases & Benchmarking

Upon official publication of our research paper, this repository will expand into a comprehensive benchmark suite. Planned updates include:
* Full data processing pipelines for standard UAV-to-Ground profiles.
* Additional baselines leveraging diverse **Supervised DNN configurations**.
* Comparative evaluation plots mapping out Mean Squared Error (MSE) and Bit Error Rate (BER).

---

##  Citation & Research Collaboration

If you find this framework useful or use our baseline architecture in your academic work, please consider citing our upcoming paper. 

*(Citation details and text will be updated here immediately upon publication)*
