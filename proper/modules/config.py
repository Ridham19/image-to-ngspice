import json
import os

CONFIG_PATH = r"D:\codes\ML\image_to_ngspice\proper\modules\component_specs.json"

class ComponentConfig:
    def __init__(self, path=CONFIG_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file {path} not found!")
        
        with open(path, 'r') as f:
            self.specs = json.load(f)
            
        # Generate handy lists for other modules
        self.class_names = sorted(list(self.specs.keys()))
        self.spice_prefixes = {k: v['prefix'] for k, v in self.specs.items()}

    def get_pin_count(self, class_name):
        """Returns expected number of pins for a component type."""
        return len(self.specs.get(class_name, {}).get("pins", []))

    def get_pin_names(self, class_name):
        return self.specs.get(class_name, {}).get("pins", [])

    def is_polarized(self, class_name):
        return self.specs.get(class_name, {}).get("polarized", False)

# Global Instance
cfg = ComponentConfig()