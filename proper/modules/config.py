import json
import os

# Get the directory where THIS file is located
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for json in the same folder
CONFIG_PATH = os.path.join(MODULES_DIR, "component_specs.json")

class ComponentConfig:
    def __init__(self):
        if not os.path.exists(CONFIG_PATH):
            # Fallback: Check parent folder just in case
            parent_path = os.path.join(os.path.dirname(MODULES_DIR), "component_specs.json")
            if os.path.exists(parent_path):
                self.load_specs(parent_path)
            else:
                raise FileNotFoundError(f"❌ Configuration file missing! Expected at: {CONFIG_PATH}")
        else:
            self.load_specs(CONFIG_PATH)

    def load_specs(self, path):
        with open(path, 'r') as f:
            self.specs = json.load(f)
        self.class_names = sorted(list(self.specs.keys()))

    def get_prefix(self, label):
        return self.specs.get(label, {}).get('prefix', 'X')

    def get_pin_names(self, label):
        return self.specs.get(label, {}).get('pins', [])

cfg = ComponentConfig()