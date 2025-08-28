# Channel Model (3GPP TR 38.901 Rel-19 §7.9)

This project uses the **UAV sensing scenarios** defined in **TR 38.901 Section 7.9**.

## Applicable Communication Scenarios
- Urban Micro (UMi)
- Urban Macro (UMa)
- Rural Macro (RMa)
- Suburban Macro (SMa)
- Aerial variations (UMi-AV, UMa-AV, RMa-AV as per TR 36.777)

## Sensing Target
- Outdoor UAVs (LOS/NLOS possible)
- Horizontal velocity: U(0,180 km/h)
- Vertical velocity: 0 km/h (optional vertical motion not used in current simulation)


## 3D Distribution
- Horizontal: 1–2 UAVs randomly distributed per scenario
- Vertical: Fixed height selected from {50, 100, 200, 300} m

## Physical Characteristics
- LOS/NLOS flag is used for supervised ML classification
- N=0 targets may be simulated for false alarm evaluation


## Notes
- LOS/NLOS flag can be used as labels for supervised learning models
- TRP and UE positions are not explicitly simulated;
- only UAV positions and resulting channel effects are modeled
- N=0 targets may be simulated for false alarm evaluation
