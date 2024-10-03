# Transit Electrification Future
This repository contains the code and data used in the research on transitioning conventional diesel bus fleets to zero-emission powertrains, focusing on battery electric buses (BEBs) and hybrid electric buses (HEBs). The study uses data from the Pioneer Valley Transit Authority (PVTA) and analyzes 10 fleet electrification futures under various fleet purchase strategies and multiple investment scenarios, with the goal of minimizing diesel consumption and emissions.

## Overview
In this reasearch we focuse on developing an optimization and energy modeling framework to analyze the transition from conventional diesel bus fleets to zero-emission powertrains. The following steps were undertaken to complete the research:

1. **Data collection**: 
   - Collected bus movement and refueling data from the Pioneer Valley Transit Authority (PVTA) covering October 2021 to September 2022.
   - Obtained bus fleet characteristics, including age, powertrain type, and mileage.
   
2. **Energy modeling**: 
   - Computed bus trajectories including distance, speed, and acceleration 
   - Developed a system-wide energy model calibrated to PVTA bus data to predict fuel consumption and energy usage for each powertrain type (CDB, HEB, BEB).
   
3. **Optimization framework**: 
   - Built an optimization model using Gurobi to assign powertrains to bus trips, aiming to minimize diesel consumption while considering investment and infrastructure constraints.
   
4. **Future evaluation**: 
   - Evaluated ten electrification futures considering fleet purchase strategies across different investment scenarios, analyzing diesel consumption, emissions, and costs.

5. **Results and visualization**: 
   - Generated comprehensive analysis and visualizations for fleet composition, diesel consumption, emissions, and other key metrics for each future scenario.

<p align="center">
  <img src="https://github.com/user-attachments/assets/38affa12-0477-4275-b80d-fa430a91e32f" alt="Diesel Consumption" width="400"/>
  <img src="https://github.com/user-attachments/assets/71f133b7-a909-4479-8fb7-6c42ba853a51" alt="GHG Emissions" width="400"/>
</p>

*Yearly diesel consumption and cumulative greenhouse gas emissions (ktCO₂e) over the 13-year planning horizon for the optimized futures. The percent reduction in emissions (or diesel consumption) relative to the **Status Quo** is shown in parentheses.*

## Repository Structure

| Directory       | Description                                                                               |
| --------------- | ----------------------------------------------------------------------------------------- |
| `bin/python/`   | Python scripts for data cleaning and implementing energy model and optimization framework.|
| `bin/jupyter/`  | Jupyter notebooks for analyzing model results and visualization.                          |
| `data/`         | Contains raw and cleaned data, including APC, and refueling tickets data.                 |
| `figures/`      | Visualizations and plots generated from the analysis, such as heatmaps and scatterplots.  |
| `results/`      | Output matrices, model validation results, and analysis outcomes.                         |

## Usage

1. **Energy Modeling:**  
   Use the energy models provided to estimate fuel consumption and energy usage for each powertrain type:

   ```bash
   python energy_model.py

3.  **Optimization:**  Run the optimization framework to assign powertrains to bus trips and minimize diesel consumption:
   
```bash
python powertrain-allocation.py
```

## Data Sources
- Automatic Passenger Counter (APC) Data: Used to compute bus trajectories and energy consumption.
- Refueling data: Provides timestamps and quantities of fuel refueled by buses.
- PVTA fleet data: Includes vehicle types, age, and powertrain information.

## Key Results
The study found that a mix of BEBs and HEBs can achieve an 87% electrification of the bus fleet by 2035, reducing diesel consumption by up to 45%. The results vary across different investment scenarios, with the most aggressive electrification strategies requiring substantial upgrades to charging infrastructure.

## Acknowledgments
This research was conducted as part of the Helping Obtain Prosperity for Everyone (HOPE) Program, funded by the Federal Transit Administration (FTA). We thank the Pioneer Valley Transit Authority (PVTA) for providing operational data and valuable insights.
