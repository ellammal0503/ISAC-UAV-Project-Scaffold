# Samsung EnnovateX 2025 AI Challenge Submission  

**Project Title**: Estimation of UAV parameters (range, velocity, direction of arrival) using **Monostatic Integrated Sensing and Communication (ISAC)** based on 3GPP TR 38.901 (Rel-19) §7.9 channel models.  

---


##  Description  
This repository contains the implementation of **AI-based ISAC (Integrated Sensing and Communication) framework for UAV sensing and parameter estimation**, developed as part of the **Samsung EnnovateX 2025 AI Challenge**.  


---


**Problem Statement** - Problem Statement #7
Estimation of UAV Parameters Using Monostatic Sensing in ISAC Scenario
Develop an AI-based solution using monostatic integrated sensing and communication (ISAC) to estimate UAV range, velocity, and direction of arrival, leveraging advanced signal processing and machine learning. Utilize the channel model based on 3GPP TR 38.901-j00 (Rel-19) Section 7.9 for ISAC applications. Participants are expected to design models that extract these parameters from ISAC signals under the specified channel conditions.  

**Team Name** - Solo Team 
**Team Members** - Karthick Kumarasamy  
**Demo Video Link** -  to be done 


---
**Project Structure**
-ISAC-UAV-Project-Scaffold/
- │── docs/ # Technical documentation
- │── src/ # Source code
- │── requirements.txt # Dependencies
- │── setup.py # Install script
- │── run.sh # Run pipeline
- │── LICENSE # License file
- │── README.md # This file


## Documentation
- See docs/ for detailed technical write-ups:
- Architecture
- Channel model
- Simulation
- Signal processing
- AI models
- Datasets
- Experiments


## Features
- 3GPP TR 38.901 §7.9 UAV scenario simulation
- Range-Doppler processing + CFAR detection
- MUSIC DoA estimation
- Feature extraction for ML/DL
- Baseline + Deep models
- Plotting & metrics utilities

## License
- MIT License – see LICENSE.

---

- #python3 -m venv venv
- #source venv/bin/activate

**Installation**
```bash

git clone https://github.com/ellammal0503/ISAC-UAV-Project-Scaffold.git
cd ISAC-UAV-Project-Scaffold

pip install -e .


## Usage
chmod +x run.sh
./run.sh


[INFO] Generated UAV scenario: [{'pos': (-145.84954113878, -178.08206464864807, 149.59066279986354), 'vel': (28.97654438306837, 5.555555555555555), 'size': (0.3, 0.4, 0.2), 'los': False}, {'pos': (420.2457855892241, 402.1074406308873, 157.36340785162017), 'vel': (27.092613058565828, 11.11111111111111), 'size': (1.6, 1.5, 0.7), 'los': True}]
[INFO] Channel initialized
[INFO] Echo signal generated
[INFO] Detections: [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
[DEBUG] Patched rx_signal shape: (8, 128)
[INFO] Extracted 64 features

[INFO] Training MLP model...
Epoch 1/5, Loss: 0.6836, Acc: 0.6250
Epoch 2/5, Loss: 0.4238, Acc: 1.0000
Epoch 3/5, Loss: 0.2723, Acc: 1.0000
Epoch 4/5, Loss: 0.2006, Acc: 1.0000
Epoch 5/5, Loss: 0.1769, Acc: 1.0000
[RESULT] MLP Accuracy: 1.00
[INFO] Saved MLP model -> results/models/MLP.pth

[INFO] Training CNN model...
Epoch 1/5, Loss: 0.5587, Acc: 0.6875
Epoch 2/5, Loss: 0.5169, Acc: 0.6875
Epoch 3/5, Loss: 0.4823, Acc: 0.6875
Epoch 4/5, Loss: 0.4459, Acc: 0.6875
Epoch 5/5, Loss: 0.4202, Acc: 0.6875
[RESULT] CNN Accuracy: 0.69
[INFO] Saved CNN model -> results/models/CNN.pth

[INFO] Training RNN model...
Epoch 1/5, Loss: 0.5462, Acc: 0.9062
Epoch 2/5, Loss: 0.3303, Acc: 1.0000
Epoch 3/5, Loss: 0.2491, Acc: 1.0000
Epoch 4/5, Loss: 0.2207, Acc: 1.0000
Epoch 5/5, Loss: 0.2073, Acc: 1.0000
[RESULT] RNN Accuracy: 1.00
[INFO] Saved RNN model -> results/models/RNN.pth

[INFO] Training Transformer model...
Epoch 1/5, Loss: 0.2593, Acc: 0.7969
Epoch 2/5, Loss: 0.0000, Acc: 1.0000
Epoch 3/5, Loss: 0.0000, Acc: 1.0000
Epoch 4/5, Loss: 0.0000, Acc: 1.0000
Epoch 5/5, Loss: 0.0000, Acc: 1.0000
[RESULT] Transformer Accuracy: 1.00
[INFO] Saved Transformer model -> results/models/Transformer.pth
[INFO] Saved comparison plot -> results/plots/model_comparison.png

