# LSLService

Python apps related to the LSL.

| App | Description |
|---| --- |
| `EEGReplay.py` | Replaying a previously recorded session, including both EEG data as well as the pinch markers. |
| `EEGSim.py` | Sending a simulated EEG signal to the LSL. | 
| `LiveEEGClassifier.py` | Live classification of EEG signals, built in 3 phases with the first phase collecting ground truth, and the two remaining phases testing live inference on two different finetuned modesl. | 
| `LSLLiveViewLogger.py` | The application used by the orchestrator to log EEG data, and to display a live view of the signals (filtered with a 1-45Hz bandpass filter), as well as pinch markers. 