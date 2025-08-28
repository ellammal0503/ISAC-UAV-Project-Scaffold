# Samsung EnnovateX 2025 AI Challenge Submission  

**Project Title**: Estimation of UAV parameters (range, velocity, direction of arrival) using **Monostatic Integrated Sensing and Communication (ISAC)** based on 3GPP TR 38.901 (Rel-19) §7.9 channel models.  

---


##  Description  
This repository contains the implementation of **AI-based ISAC (Integrated Sensing and Communication) framework for UAV sensing and parameter estimation**, developed as part of the **Samsung EnnovateX 2025 AI Challenge**.  


---

# Samsung EnnovateX 2025 AI Challenge Submission  


**Problem Statement** - Problem Statement #7
Estimation of UAV Parameters Using Monostatic Sensing in ISAC Scenario
Develop an AI-based solution using monostatic integrated sensing and communication (ISAC) to estimate UAV range, velocity, and direction of arrival, leveraging advanced signal processing and machine learning. Utilize the channel model based on 3GPP TR 38.901-j00 (Rel-19) Section 7.9 for ISAC applications. Participants are expected to design models that extract these parameters from ISAC signals under the specified channel conditions.  

**Team Name** - Solo Team 
**Team Members** - Karthick Kumarasamy  
**Demo Video Link** -  to be done 


---
**Project Structure**
- ISAC-UAV/
-- │── docs/ # Technical documentation
-- │── src/ # Source code
-- │── requirements.txt # Dependencies
-- │── setup.py # Install script
-- │── run.sh # Run pipeline
-- │── LICENSE # License file
-- │── README.md # This file


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
- MIT License
- – see LICENSE.

---

#python3 -m venv venv
#source venv/bin/activate

**Installation**
```bash
git clone https://github.com/ellammal0503/ISAC-UAV-Project-Scaffold.git
cd ISAC-UAV-Project-Scaffold
pip install -e .
#chmod +x run.sh
## Usage
./run.sh

