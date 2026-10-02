# VIBE — Virtual Input via Brain-computer interfacE

Original title: *Practical Input Detection using EEG for Virtual Reality*

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
├── DataAnalysis/               # Data anaylsis, and model creation for pinch detection.
├── Experiment/                 # The Unity project for the experiment.
├── ExperimentOrchestrator/     # Python utility to automate the data collection.
├── LSLService/                 # Data collection and live inference utilities.
├── MyoDetector/                # (discontinued) an early prototype for a Myo armband pinch detection.
├── TechDemo/                   # Test environment for the final experiment.
└── weekly_reports/             # Brief resume of weekly progres.
```