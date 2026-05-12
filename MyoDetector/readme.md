# MyoDetector

This small app, powered by [dl-myo](https://github.com/iomz/dl-myo), is a very basic pinch detector for Thalmic Myo 
Armband.

## Getting Started

Before you start, pair the Myo with your computer - wearing the armband will turn it on, and it will then be 
discoverable in your Bluetooth settings.

```shell
# Clone the dl-myo repository and copy the myo package.
git clone git@github.com:iomz/dl-myo.git src/dl-myo
mv src/dl-myo/myo src/myo
rm -rf src/dl-myo
# Calibrate
uv run .\src\pinch_detector.py calibrate
# Detect and publish to LSL
uv run .\src\pinch_detector.py detect
```

The script will now detect any pinch events and send them to the labstreaminglayer as markers. The LSL integration can 
be deactivated with the additional parameter `--dry-run`.