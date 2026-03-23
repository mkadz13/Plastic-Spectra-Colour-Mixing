Spectral color mixing optimizer for waste plastic 3D printing. Takes spectral reflectance data from recycled plastics and calculates optimal mixing ratios to reproduce a target color.

## Setup:
git clone https://github.com/mkadz13/Plastic-Spectra-Colour-Mixing.git
cd Plastic-Spectra-Colour-Mixing
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt

## Desktop Application:
cd Plastic-Spectra-Colour-Mixing
python run_desktop.py

## How to Use Program:
1. Select a **target color** (the color you want to make)
2. Check the **ingredient colors** (the waste plastics you have)
3. Pick a **solver** (Nelder-Mead recommended)
4. Set **total grams**
5. Click **Optimize**
