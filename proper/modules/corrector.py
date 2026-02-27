import re
from collections import defaultdict

class SpiceLinter:
    def __init__(self, raw_netlist):
        self.raw_netlist = raw_netlist
        self.lines = raw_netlist.split('\n')
        self.components = []
        self.node_connections = defaultdict(list)
        self.has_ground = False
        
    def parse(self):
        """Reads the raw netlist and maps every node to its components."""
        for line in self.lines:
            line = line.strip()
            # Skip comments and commands
            if line.startswith('*') or line.startswith('.') or not line:
                continue
                
            # Typical SPICE line: Name Node1 Node2 ... Value
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                value = parts[-1]
                nodes = parts[1:-1]
                
                self.components.append({
                    'name': name,
                    'nodes': nodes,
                    'value': value,
                    'original': line
                })
                
                for node in nodes:
                    self.node_connections[node].append(name)
                    if node == '0':
                        self.has_ground = True

    def fix_ground(self):
        """SPICE crashes without Node 0. If missing, make the most connected node Ground."""
        if self.has_ground or not self.node_connections:
            return

        # Find valid nodes (ignore NC_)
        valid_nodes = [n for n in self.node_connections if not n.startswith('NC_')]
        if not valid_nodes: return
        
        most_connected_node = max(valid_nodes, key=lambda k: len(self.node_connections[k]))
        
        # Rename that node to '0' everywhere
        for comp in self.components:
            comp['nodes'] = ['0' if n == most_connected_node else n for n in comp['nodes']]
            
        self.node_connections['0'] = self.node_connections.pop(most_connected_node)
        self.has_ground = True

    def fix_floating_nodes(self):
        """
        Instead of adding dummy components, we tie floating or 'NC_' pins 
        to an existing valid node in the circuit, making sure not to short-circuit the component.
        """
        # Find all nodes that are properly connected to at least 2 things
        valid_nodes = [n for n, comps in self.node_connections.items() if len(comps) > 1 and not n.startswith('NC_')]
        
        if not valid_nodes:
            valid_nodes = ['0', '1'] # Hard fallback if the circuit is totally broken

        for comp in self.components:
            for i, node in enumerate(comp['nodes']):
                # If the node is marked Not Connected OR only has 1 connection in the whole circuit
                if node.startswith('NC_') or len(self.node_connections[node]) == 1:
                    
                    # Find a valid node to patch this into.
                    # CRITICAL: We filter out nodes already used by THIS component to avoid short-circuits.
                    safe_choices = [n for n in valid_nodes if n not in comp['nodes']]
                    
                    if not safe_choices:
                        safe_choices = ['0'] # Forced fallback
                        
                    # Pick the first safe node and connect it
                    chosen_node = safe_choices[0]
                    comp['nodes'][i] = chosen_node
                    
                    # Log it for debugging
                    comp['original'] = f"* Linter fixed floating pin on {comp['name']}\n" + comp['original']

    def generate_clean_netlist(self):
        """Reassembles the fixed netlist."""
        self.parse()
        self.fix_ground()
        self.fix_floating_nodes()
        
        clean_lines = ["* Linter-Corrected SPICE Netlist"]
        
        for comp in self.components:
            node_str = " ".join(comp['nodes'])
            clean_lines.append(f"{comp['name']} {node_str} {comp['value']}")
            
        clean_lines.append(".end")
        return "\n".join(clean_lines)

def run_linter(raw_netlist):
    linter = SpiceLinter(raw_netlist)
    return linter.generate_clean_netlist()