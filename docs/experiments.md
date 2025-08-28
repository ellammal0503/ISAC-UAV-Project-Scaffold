# Experiments & Results

## Goals
- Evaluate UAV parameter estimation accuracy (range, velocity, DoA)
- Compare traditional signal processing with AI-based models (MLP, CNN, RNN, Transformer)
- Assess model generalization on synthetic UAV scenarios (LOS/NLOS).


## Metrics
- RMSE for range, velocity, DoA
- Probability of detection (Pd) vs false alarm (Pfa)
- OSPA metric: for multi-target estimation accuracy
- Classification Accuracy: for discrete labels if ML models classify target properties

## Experiments
1. **Baseline (signal processing only)**
   - RDM + CFAR + MUSIC
   - Direct estimation of UAV parameters
2. **Feature-based ML/Hybrid Approach**
   - Extracted features from signal processing (RDM + DoA) → train ML/DL models
    - MLP
    - CNN
    - RNN
Transformer
3. **End-to-end DL**
   - CNN/Transformer trained directly on raw RAI cubes for regression/classification of UAV parameters

## Results (Placeholder)
- Accuracy, loss curves, and comparison plots for each model saved under results/plots/
- Model checkpoints saved under results/models/
- To be updated after complete training & evaluation on synthetic scenarios
