import argparse
import sys
import os

# Add 'modules' to path so imports work anywhere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.gui import run_main_gui
import modules.labeling as labeler

def main():
    parser = argparse.ArgumentParser(description="Ridham19-AI: Circuit to SPICE")
    parser.add_argument("--label", action="store_true", help="Launch the Manual Labeling/Dataset Tool")
    args = parser.parse_args()

    if args.label:
        print("Launching Manual Labeling Tool...")
        labeler.run_labeler()
    else:
        print("Launching Main GUI...")
        run_main_gui()

if __name__ == "__main__":
    main()