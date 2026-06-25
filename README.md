# OlfactoBot AI

OlfactoBot is a runnable Python prototype of a mobile AI-powered artificial nose. It simulates sensor arrays, analytical chemistry-like volatile features, biological meaning labels, autonomous plume mapping, machine-learning training, and robot decision-making for health, food, and environmental settings.

## What it does

The program simulates a robot that can:

- collect electronic-nose sensor-array data
- correct sensor drift and environmental artefacts
- simulate volatile chemical mixtures
- learn chemical meaning using machine learning
- classify samples as safe background, food spoilage, infection-like VOC, or environmental hazard
- estimate uncertainty
- map a chemical plume in a 2D environment
- decide whether to patrol, move closer, collect a sample, retreat, or alert a human operator
- save figures, tables, model files, and a mission report

## Quick start

```bash
cd olfactobot_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python olfactobot.py --mode full
```

## Useful commands

Train the AI model only:

```bash
python olfactobot.py --mode train
```

Run one robot mission using the trained model:

```bash
python olfactobot.py --mode mission
```

Generate data, train model, run mission, and save all outputs:

```bash
python olfactobot.py --mode full
```

Outputs are saved in:

```text
outputs/
models/
```

## Note

This is a simulation and prototype. It is intended for technical demonstration, ARIA interview preparation, and early research design. It is not a validated diagnostic, food-safety, or environmental-safety system.
