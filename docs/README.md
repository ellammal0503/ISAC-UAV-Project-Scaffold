# ISAC-UAV Project

This project implements an **AI-based ISAC (Integrated Sensing and Communication)** framework for UAV sensing and parameter estimation.It combines synthetic UAV scenario simulation, signal processing, and machine learning/deep learning models to estimate UAV range, velocity, and direction of arrival (DoA) in monostatic radar setups. 

## Problem Statement
Estimate UAV **range**, **velocity**, and **direction of arrival (DoA)** in monostatic ISAC scenarios using:
- Advanced signal processing (RDM, CFAR, MUSIC/ESPRIT)
- AI Models: MLP, CNN, RNN, Transformer
- Channel models based on **3GPP TR 38.901 Rel-19 Section 7.9**

## Repository Layout
- `docs/` → All technical documentation (this folder)
- `src/` → Source code (simulation, processing, ML models, training)
- `requirements.txt` → Dependencies
- `setup.py` → Installable package definition
- `run.sh` → Script to run end-to-end pipeline

## Outputs
- Trained Models → Stored in results/models/ (MLP, CNN, RNN, Transformer)
- Generated Datasets → Synthetic UAV ISAC dataset with features and labels (range, velocity, DoA)
- Plots & Metrics → Training loss/accuracy, RDM, DoA spectrum, performance comparison saved under results/plots/
- Attribution → Based on open-source implementations and 3GPP TR 38.901 standards.
