# MyoDetector

This small app, powered by [dl-myo](https://github.com/iomz/dl-myo), is a very basic pinch detector for Thalmic Myo 
Armband.

## Getting Started

1. Clone above repo into the `src` folder: `git clone git@github.com:iomz/dl-myo.git src/dl-myo`
2. Pair the Myo with your computer - wearing the armband will turn it on, and it will then be discoverable in your 
   Bluetooth settings.
3. Run the calibration procedure: `uv run .\src\pinch_detector.py calibrate`
4. Start detecting pinches:  `uv run .\src\pinch_detector.py detect`

The script will now detect any pinch events and send them to the labstreaminglayer as markers. The LSL integration can 
be deactivated with the additional parameter `--dry-run`.