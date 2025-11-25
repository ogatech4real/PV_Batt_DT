## Important Notice

This repository accompanies an IEEE Access manuscript currently under peer review.  
It has been made publicly accessible **exclusively for reviewer and editorial evaluation**.  
No part of this work, including the code, data, figures, or manuscript text, may be copied, reused, or cited until the article has been formally accepted and published.  

For questions, collaboration requests, or clarification, please get in touch with the corresponding author directly at **hello@adewaleogabi.info, ogabi.adewale@gmail.com**.

---

# Materials-Aware Digital Twin for Solar–Battery Systems  
**A reproducible framework for lifecycle-aware optimisation of photovoltaic (PV) and battery energy storage systems**

---

## Overview

This repository contains the complete implementation of the **Materials-Aware Digital Twin (MAT-DT)** framework proposed in the IEEE Access manuscript:  
**“Materials-Aware Digital Twin for Solar–Battery Systems: Integrating Equipment Ageing into Smart Energy Optimisation.”**

The project integrates physical degradation models for both batteries and photovoltaic (PV) modules directly into an optimisation-based control system.  
It demonstrates how digital twins can move beyond monitoring to perform **active lifecycle management**, balancing short-term cost efficiency with long-term equipment health and carbon reduction.

---

## Research Motivation

Conventional energy management systems optimise PV–battery operation for short-term savings (e.g., tariff arbitrage) while ignoring battery wear and PV derating.  
Over time, this leads to reduced capacity, efficiency loss, and hidden lifecycle costs.

This project introduces a **materials-aware digital twin** that:
- Models real degradation processes (battery and PV ageing).  
- Uses forecasts and optimisation to inform daily operational decisions.  
- Quantifies trade-offs between cost, carbon emissions, and asset lifetime.  
- Provides a reproducible simulation platform ready for hardware-in-the-loop (HIL) integration.

---

## Core Features

- **Reduced-order ageing models** for battery and PV degradation (calendar + cycle ageing; temperature and environmental stress effects).  
- **Rolling-horizon optimisation** using [Pyomo](https://www.pyomo.org/) for cost–carbon–lifecycle balancing.  
- **Forecasting module** for solar generation, load demand, and tariffs.  
- **Bootstrap validation and Pareto analysis** for statistical confidence and trade-off exploration.  
- **Modular design**—each script handles a specific task (generation, modelling, optimisation, evaluation).  
- **Reproducible results** aligned with figures and tables presented in the IEEE Access paper.

---

## Methodology Summary

The framework follows a structured digital twin architecture linking six key modules:

1. **Data Generation:** Creates standardised 15-minute datasets representing PV, load, and tariff profiles.  
2. **Degradation Modelling:** Quantifies battery capacity fade and PV derating using empirical relationships.  
3. **Forecasting:** Predicts near-term irradiance, load, and tariffs using statistical or ML methods.  
4. **Optimisation:** Solves a rolling-horizon problem to minimise total cost, carbon emissions, and degradation.  
5. **Control Execution:** Applies the first decision at each step, updating system states iteratively.  
6. **Evaluation:** Computes KPIs (economic, lifecycle, environmental) and performs statistical validation.

---

## The workflow can be executed entirely via:

  ``bash
python main.py
All figures and tables in the paper can be regenerated from the results/ and figs/ directories.

---

##  Key Performance Indicators (KPIs)

  1. **Economic**: Annual electricity cost, arbitrage revenue, peak import reduction, LCOS.
  2. **Lifecycle**: Battery capacity fade (%), equivalent full cycles, PV performance ratio decline.
  3. **Environmental**: Avoided CO₂ emissions (kgCO₂e).
  4. **Reliability**: Demand served (%), reserve margin compliance.

---

## Simulation Results (Summary)

  - Battery wear reduction by 10–20% compared to the cost-only control.
  - PV degradation slowed by ≈0.1%/year.
  - Cost difference < 3% from baseline while improving lifetime economics.
  - Results statistically validated via bootstrapping and Pareto trade-off analysis.

---

## Implementation Environment

- Language: Python 3.10+
- Optimisation Engine: Pyomo (CBC solver)
- Key Libraries: Pandas, NumPy, Matplotlib, Scikit-learn, Statsmodels
- Runtime: Tested on Windows 11 and Ubuntu 22.04
- Hardware: 8GB RAM minimum; optional Raspberry Pi deployment for real-time control

---

## Hardware Integration (Optional)

The framework is ready for hardware-in-the-loop (HIL) validation.
  - A typical setup includes:
  - Raspberry Pi as edge controller running the optimisation engine.
  - Arduino UNO WiFi as sensor/actuator interface for voltage, current, and temperature measurements.
  - Communication: MQTT, Modbus, or REST API between the physical system and the digital twin.
  - This setup enables real-time testing of lifecycle-aware control strategies.


---

## Citation

If you use this repository, please cite the following paper:

A. Ogabi, G. Aggarwal, and A. Alabi,
“Materials-Aware Digital Twin for Solar–Battery Systems: Integrating Equipment Ageing into Smart Energy Optimisation,”
IEEE Access, 2025. DOI: to be assigned.
GitHub Repository



---

## License

This repository is distributed under the MIT License.
You are free to use, modify, and distribute this work with appropriate credit.


---

## Contact

- Author: Adewale Ogabi
- Affiliation: School of Computing, Engineering, and Digital Technologies, Teesside University, UK
- Email: hello@adewaleogabi.info, ogabi.adewale@gmail.com

Repository: https://github.com/ogatech4real/Material_Aware_Digital_Twin


