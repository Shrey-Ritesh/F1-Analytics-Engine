import json
import os

def load_regulations():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    reg_file = os.path.join(project_root, 'model', 'strategy_optimizer', 'regulations.json')
    with open(reg_file, 'r') as f:
        return json.load(f)

def get_circuit_rules(circuit, regulations=None):
    if regulations is None:
        regulations = load_regulations()
        
    rules = regulations['default_rules'].copy()
    overrides = regulations['circuit_overrides'].get(circuit, {})
    
    for k, v in overrides.items():
        rules[k] = v
        
    return rules

def validate_strategy(strategy, circuit, total_laps, regulations=None):
    rules = get_circuit_rules(circuit, regulations)
    violations = []

    # Rule 1: minimum stops
    if strategy['n_stops'] < rules['min_stops']:
        violations.append(
            f"min stops violation: needs {rules['min_stops']}, "
            f"has {strategy['n_stops']}"
        )

    # Rule 2: minimum compounds
    if len(set(strategy['compounds'])) < rules['min_compounds']:
        violations.append(
            f"min compounds violation: needs {rules['min_compounds']}, "
            f"has {len(set(strategy['compounds']))}"
        )

    # Rule 3: max laps per set (Qatar)
    if rules.get('max_laps_per_set'):
        pit_laps = strategy['pit_laps']
        if len(pit_laps) == 0:
            stints = [total_laps]
        else:
            stints = (
                [pit_laps[0]] +
                [pit_laps[i+1] - pit_laps[i] for i in range(len(pit_laps)-1)] +
                [total_laps - pit_laps[-1]]
            )
        for i, length in enumerate(stints):
            if length > rules['max_laps_per_set']:
                violations.append(
                    f"stint {i+1} is {length} laps — "
                    f"exceeds {rules['max_laps_per_set']} lap max"
                )

    return {
        'valid': len(violations) == 0,
        'violations': violations
    }

def print_circuit_rules(circuit):
    rules = get_circuit_rules(circuit)
    print(f"\n=== Rules: {circuit} ===")
    print(f"  Min stops:        {rules['min_stops']}")
    print(f"  Min compounds:    {rules['min_compounds']}")
    print(f"  Max laps per set: {rules.get('max_laps_per_set', 'None')}")
