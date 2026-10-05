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

## The experiment

### Starting a session

To run the experiment, we need Unity, and the Experiment Orchestrator.

1. Calibrate the EEG with the proprietary software.
2. Mount the VR headset, and perform eye callibration.
3. Check that the EEG hetset is still well placed, and close the software (the COM port must not be used by any other program before starting the orchestrator)
4. Open the Unity Project and run the `Experiment1` scene in the editor mode. The scene starts in the "waiting room" for a session selection.
5. Start the respective session with `Ctrl + [session number]`. Unity now waits for the experiment orchestrator.
6. Start ExperimentOrchestrator: `uv run .\ExperimentOrchestrator.py [subject id]`
7. Confirm that eye calibration and EEG calibration was done.
8. The instructions and tutorial automatically starts. After the tutorial, a blank screen is shown.
9. After answering any remaining questions, press `Ctrl + c` to start the main experiment.
10. During the experiment, manual inputs (space bar) must be done for hinted pinches. 
11. At the end, stop the experiment by closing the EEG signal preview. 

### Starting a live inference session

1. Perform the above steps until 7, starting session 3 with `Ctrl + 3`.
2. Now start the live inference with the following command below (models can be generated in `DataAnalysis/15_classification.ipynb`)
3. Wait for the experiment to start. After the tutorial, confirm with `Ctrl + C`
4. After the next phase is ready, confirm continuation with `Ctrl + C`

In phase 2 and 3 of the session, the Leap Motion can be kept focussed on the hand to collect ground truth data that does not trigger a stimulus activation.

```
uv run .\LiveEEGClassifier.py [path/to/model/1] [path/to/model/2] --subject [subject number of the subject to do inference]
```

### Unity Keyboard Shortcuts

- `Ctrl + C`: continue when Unity is waiting for confirmation (can also be used when we just want to test without the orchestrator, in the "waiting room")
- `Spacebar`: manual pinch input
- `Ctrl + 1`: start session 1 (in "waiting room")
- `Ctrl + 2`: start session 2 (in "waiting room")
- `Ctrl + 3`: start session 3 (in "waiting room", session 3 requires additionally the LiveEEGClassifier)
- `Ctrl + k`: pause at next possible location
- `Ctrl + j`: skip current block, executed at next possible location
- `Ctrl + l`: repeat current block, executed at next possible location

### Dependencies for the Unity Project

Most dependencies are internal to the unity project. The following dependencies are required for VR to work. 

- Steam VR
- Vive Console for Steam VR (Steam)