# VIBE — Virtual Input via Brain-computer interfacE

Original title: *Practical Input Detection using EEG for Augmented Reality*

**Master's Thesis | ETH Zürich | Dept. of Computer Science**  
Supervisor: Prof. Christian Holz

---

## Overview

This thesis investigates whether a thumb-index finger pinch can be detected from EEG signals reliably enough to serve as a practical selection trigger in XR. The project combines VR task design, multimodal sensing, EEG analysis, and event detection to explore EEG as a viable input modality — potentially replacing camera-based hand tracking.

---

## Research Questions

- Can a thumb-index pinch be detected from EEG robustly enough to serve as a practical selection event in XR?
- To what extent do visual, auditory, and tactile feedback contribute to the EEG signatures associated with pinch events?
- How well can EEG-based pinch-event detection generalize across users, sessions, and feedback conditions?

---

## Repository Structure

```
├── data/               # Raw and preprocessed EEG recordings (excluded from version control)
├── TechDemo/           # Unity XR task application and event logging
├── LSLService/         # Lab Streaming Layer related code for data gathering
└── results/            # Exported figures, tables, and model outputs
```

---

## Setup

```bash
git clone https://github.com/baeriswyln/VIBE.git
cd VIBE
```

---

## Experiment

Participants perform pinch-based target selection in a Unity XR scene under four feedback conditions:

```
todo
```

EEG, hand tracking (Leap Motion), and VR event markers are synchronized via **Lab Streaming Layer (LSL)**.

---

## Models

- **Baseline:** Feature extraction + LDA/SVM
- **Deep learning:** EEGNet
- **Stretch goal:** EEG Conformer + Euclidean Alignment for cross-subject transfer

---

## Status

- [x] Literature review
- [ ] Experimental setup & Unity app
- [ ] Data recording (~6 participants)
- [ ] EEG preprocessing & analysis
- [ ] Model training & evaluation
- [ ] Real-time proof of concept *(stretch goal)*
- [ ] Thesis writing
