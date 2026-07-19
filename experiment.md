# Experiment

## Setup

## Tutorial

Before starting the actual measurements, the participant will go through a tutorial to reduce learning effects during the measurements.

## Tasks

### Free variables

The following variables are present in the experiment

- Virtual hand visible
- Visual feedback
- Auditory feedback
- Induced errors (ignore two pinches to provoke error related potentials)
- Hinted pinch (small indication of a pinch without a full movement, wizard-of-oz setup)
- Observe (some stimuli must only be observed without pinching)
- Choice (the user gets to chose which stimuli to select)

### Stimuli setups

There are three different stimuli setups:

- `Static_1_Stimuli`: Exactly one stimulus is spawned at the center of the view port. No eye/head movement is required, reducing movement induced EEG signals.
- `Static_4_Stimuli`: Four stimuli are spawned on an imaginary circle around the center of the view port, regularly spaced from eachother. This requres the participant to move their eyes. 
- `Random_6_Stimuli`: Six stimuli are spawned at random positions in the view port with a minimum space between eachother such that eye and slight head movements will be requried. 

### Task details

The following table lists the tasks with their free variables that the participant will encounter. Each round contains a total of 12 events. The task `Static_1_Stimuli` will always be the first and last task. All other tasks appear exactly once in a random order to prevent a data skew originating from a potential learning effect. 

| Stimuli setup | Virtual Hand | Visual | Auditory | Induced errors | Hinted pinch | Observe | Choice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static_1_Stimuli |  x | x | x | | | |
| Static_4_Stimuli | x | x | x |  |  |  |  |
| Static_4_Stimuli | x |  | x |  |  |  |  |
| Static_4_Stimuli | x | x |  |  |  |  |  |
| Static_4_Stimuli |  | x | x |  |  |  |  |
| Static_4_Stimuli |  | x |  |  |  |  |  |
| Static_4_Stimuli |  |  | x |  |  |  |  |
| Static_4_Stimuli |  | x | x | x |  |  |  |
| Static_4_Stimuli |  | x | x | x | x |  |  |
| Static_4_Stimuli |  | x | x |  |  | x |  |
| Static_4_Stimuli |  | x | x |  |  |  | x |
| Random_6_Stimuli | x | x | x |  |  |  |  |
| Random_6_Stimuli |  | x | x | x |  |  |  |
| Random_6_Stimuli |  | x |  |  |  |  |  |
| Random_6_Stimuli |  |  | x |  |  |  |  |
| Random_6_Stimuli |  | x | x | x | x |  |  |
| Random_6_Stimuli |  | x | x |  |  | x |  |
