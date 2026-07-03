# C-TRIP: Estimating Phenotypes from Localizer MRI through Multi-Modal Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
Official PyTorch implementation of **Opportunistic Cardiac Health Assessment: Estimating Phenotypes from Localizer MRI through Multi-Modal Representations**, accepted at **MICCAI2026**.

C-TRIP (Cardiac Tri-modal Representations for Imaging Phenotypes) is a multimodal contrastive learning framework that aligns Cardiac MRI (Localizer & Cine), Electrocardiograms (ECG), and clinical Tabular phenotypes into a shared latent space and predicts CPs using localizer images as an opportunistic alternative to CMR. 

![C-TRIP Architecture Overview](./figs/ctrip_diagram.png) 

##  Installation

**1. Clone the repository**
```bash
git clone [https://github.com/basrazey/C-TRIP](https://github.com/basrazey/C-TRIP)
cd C-TRIP