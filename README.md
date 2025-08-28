# ISAC-UAV 🚀

Estimation of UAV parameters (range, velocity, direction of arrival) using **Monostatic Integrated Sensing and Communication (ISAC)** based on 3GPP TR 38.901 (Rel-19) §7.9 channel models.

## 📂 Project Structure
ISAC-UAV/
│── docs/ # Technical documentation
│── src/ # Source code
│── requirements.txt # Dependencies
│── setup.py # Install script
│── run.sh # Run pipeline
│── LICENSE # License file
│── README.md # This file



## 🔧 Installation
```bash
git clone https://github.com/yourusername/ISAC-UAV.git
cd ISAC-UAV
pip install -e .

## 🔧 Usage
./run.sh


📑 Documentation
See docs/ for detailed technical write-ups:
Architecture
Channel model
Simulation
Signal processing
AI models
Datasets
Experiments
🛠 Features
3GPP TR 38.901 §7.9 UAV scenario simulation
Range-Doppler processing + CFAR detection
MUSIC DoA estimation
Feature extraction for ML/DL
Baseline + Deep models
Plotting & metrics utilities

📜 License
MIT License – see LICENSE.

---

✅ With this, your repo is **fully bootstrapped**: code, docs, and runnable setup.  

Do you want me to now create the **sample workflow in `src/main.py`** that ties simulation → processing → ML together, so evaluators can run and see an end-to-end demo

#python3 -m venv venv
#source venv/bin/activate