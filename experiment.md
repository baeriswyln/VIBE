# Experiment

## Setup

## Tutorial

Before starting the actual measurements, the participant will go through a tutorial to reduce learning effects during the measurements.

## Independent variables

The following variables are present in the experiment

- Stimuli setup: static, eye movement, head movement
- Feed forward: visual representation of the tracked hand shown in game space - real-time feedback of the participants hand pose.
- Feedback types: visual, auditory
- Pich type: hard vs. hinted pinch (small indication of a pinch without a full movement, wizard-of-oz setup)
- Induced errors (ignore two pinches to provoke error related potentials)
- Observe (some stimuli must only be observed without pinching, minimum dwell time longer than .5s)
- Choose (the user gets to chose which stimuli to select, random choice by the participant, no color)

### Stimuli setups

There are three different stimuli setups:

- `Static_1`: Exactly one stimulus is spawned at the center of the view port. **No eye/head movement** is required, reducing movement induced EEG signals.
- `Static_4`: Four stimuli are spawned on an imaginary circle around the center of the view port, regularly spaced from eachother. This requres the **eye movement required** by the participant. 
- `Random_6`: Six stimuli are spawned at random positions in the view port with a minimum space between eachother such that **eye and slight head movements will be requried**. 

### Feedback types

The experiments will have three different feedback types.

- `Virtual hand`: A virtual representation of the participant's hand is shown, replicating their hand's posture.
- `Visual`: The stimulus will change its appearance when clicked.
- `Auditory`: A sound is played when the stimulus is clicked.

### Pinch Type

- `Hard pinch`: a real pinch, used as a default
- `Hinted pinch`: the participant only makes a slight gesture, that will be observed by the experimenter playing wizard-of-oz.

### Block details

The experiment will be separated in two sessions. The following table lists the session's blocks with their independent variables that the participant will encounter. 

#### Session 1

In the first session, each block contains a total of 20 trials. The block `Static_1_Stimuli` will always be the first and last block. All other blocks appear exactly once in a random order to prevent a data skew originating from a potential learning effect. 

The full experiment (including setup and tutorial time) is expected to take approximately 1h. 

| Stimuli setup | #trials |  Virtual Hand | Visual | Auditory | #Induced errors | Hinted pinch | #Observe  |
| ---------------- | --- | --- | --- | --- | --- | --- |
| Static_4 | 15 | x | x | x |   |  |  |
| Static_1 | 15 | x | x | x |   |   |
| Static_4 | 15 | x |   | x | 1  |  |  |
| Static_4 | 15 | x | x |   | 1  |  |  |
| Static_4 | 15 |   | x | x | 3  |  |  |
| Static_4 | 15 |   | x | x | 5  | x |  |
| Static_4 | 15 |   | x | x |   |  | 10 |
| Random_6 | 15 |   | x |   | 1 |  |  |
| Random_6 | 15 |   |   | x | 3 |  |  |
| Random_6 | 15 |   | x | x | 5 | x |  |
| Random_6 | 15 |   | x | x |   |  | 10 |
| Static_4 | 15 | x | x | x |   |  |  |

#### Session 2

The second session is shorter and will be a "control" session. Each block contains a total of 10 trials.

The full experiment (including setup and tutorial time) is expected to take approximately 30min.

| Stimuli setup | #trials | Virtual Hand | Visual | Auditory | #Induced errors | Hinted pinch | Choose | Delayed Feedback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static_4 | 10 | x | x | x |   |   |   |   |
| Static_4 | 10 |   | x | x |   |   | x |   |
| Static_4 | 10 |   | x | x |   |   | x | x |
| Static_4 | 10 |   | x | x | 3 | x | x |   |
| Static_4 | 10 |   | x | x | 3 | x | x | x |
| Random_6 | 10 |   | x |   |   |   | x |   |
| Random_6 | 10 |   |   | x |   |   | x | x |
| Static_4 | 10 | x | x | x |   |   |   |   |

## Hypotheses

- Pinches where the participant selects a *highlighted stimulus* can be observed through electrodes C3, C4, Cz that are placed on the *sensorimotor cortex*. The singal in C3 is the strongest, as the right hand will be used. 
- Pinches where the participant selects a *stimulus at their own choosing* can be observed through electrodes C3, C4, Cz. 
- Static_1_Stimuli (no head and eye movements) give the clearest signals, as few to no movement-based artifacts are present in the signal.
- More feed forward/feedback modes result in clearer signals in the stimulus-preceding negativity (SPN). [Towards an Eye-Brain-Computer Interface: Combining Gaze with the Stimulus-Preceding Negativity for Target Selections in XR (Reddy, Proulx, Hirshfield, Ries)](https://dl.acm.org/doi/10.1145/3613904.3641925)
- `Observe` events can be distinguished from `pinch` events through a reduced variability in signal strength on electrodes C3, C4, Cz.
- Induced errors provoke error related potentials (observable on Cz and Fz). [A Neural System for Error Detection and Compensation (1993, Gehring, Gross, Donchin)](https://journals.sagepub.com/doi/10.1111/j.1467-9280.1993.tb00586.x).
- Hinted pinches result in a reduced signal variability on electrodes C3, C4, Cz compared to "hard" pinches. 