import os
import sys
import subprocess

# Setup Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MAP_OPTIONS = {
    "1": ("bgc_full",  "BGC Full"),
    "2": ("bgc_core",  "BGC Core"),
    "3": ("4by4_map",  "4x4 Grid"),
}

LEVEL_OPTIONS = {
    "1": "low",
    "2": "med",
    "3": "high",
    "4": "highEnough",
}

def prompt_map() -> str:
    print("\nWhich map do you want to generate routes for?")
    print("  1. Full  (bgc_full)")
    print("  2. Core  (bgc_core)")
    print("  3. Grid  (4by4_map)")
    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice in MAP_OPTIONS:
            folder, label = MAP_OPTIONS[choice]
            print(f"Selected: {label}")
            return folder
        print("Invalid choice. Please enter 1, 2, or 3.")

def prompt_levels() -> list:
    print("\nWhich traffic levels do you want to generate?")
    print("  1. Low")
    print("  2. Med")
    print("  3. High")
    print("  4. HighEnough")
    print("  5. All (low, med, high, highEnough)")
    while True:
        choice = input("Enter choice [1/2/3/4/5]: ").strip()
        if choice == "5":
            return list(LEVEL_OPTIONS.values())
        if choice in LEVEL_OPTIONS:
            return [LEVEL_OPTIONS[choice]]
        print("Invalid choice. Please enter 1–5.")

def prompt_assignment_method() -> str:
    print("\nRouting assignment method:")
    print("  1. SUE  (Stochastic User Equilibrium) [default]")
    print("  2. incremental")
    while True:
        choice = input("Enter choice [1/2, or press Enter for default]: ").strip()
        if choice in ("", "1"):
            return "SUE"
        if choice == "2":
            return "incremental"
        print("Invalid choice.")

def prompt_max_alternatives() -> str:
    while True:
        val = input("\nMax route alternatives per vehicle? [default: 5]: ").strip()
        if val == "":
            return "5"
        if val.isdigit() and int(val) >= 1:
            return val
        print("Please enter a positive integer.")

def run_marouter(scenario_dir, map_file, level, assignment_method, max_alternatives):
    print(f"\n{'='*50}")
    print(f"   GENERATING WARDROP ROUTES: {level.upper()} DENSITY")
    print(f"{'='*50}")

    trips_filename = f"trips_{level}.xml"
    routes_filename = f"wardrop_routes_{level}.rou.xml"
    trips_file = os.path.join(scenario_dir, trips_filename)
    output_file = os.path.join(scenario_dir, routes_filename)

    if not os.path.exists(trips_file):
        print(f">>> ERROR: Input file missing: {trips_file}")
        print(f">>> SOLUTION: Generate trips_{level}.xml with randomTrips.py first and place it in {scenario_dir}.")
        return False

    if not os.path.exists(map_file):
        print(f">>> ERROR: Network file missing: {map_file}")
        return False

    vtypes_file = os.path.join(scenario_dir, "vtypes.add.xml")

    cmd = [
        "marouter",
        "-n", map_file,
        "-r", trips_file,
        "-o", output_file,
        "--assignment-method", assignment_method,
        "--max-alternatives", max_alternatives,
        "--ignore-errors", "true",
        "--verbose", "true"
    ]

    # If a vType definition file exists, tell marouter to use the passenger
    # vType so it respects allow/disallow attributes and skips pedestrian-only edges.
    if os.path.exists(vtypes_file):
        cmd += ["-a", vtypes_file, "--vtype", "thesis_car"]

    print(f">>> Running SUMO marouter on {trips_filename}...")
    try:
        subprocess.run(cmd, check=True)
        print(f">>> SUCCESS: Routes saved to {output_file}")
        print(f">>> READY: {level.upper()} traffic scenario generated.\n")
        return True
    except subprocess.CalledProcessError:
        print(f">>> ERROR: marouter failed for {level}.\n")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("   WARDROP ROUTE GENERATION")
    print("="*50)

    map_folder        = prompt_map()
    levels            = prompt_levels()
    assignment_method = prompt_assignment_method()
    max_alternatives  = prompt_max_alternatives()

    scenario_dir = os.path.join(ROOT_DIR, map_folder)
    map_file     = os.path.join(scenario_dir, "final_map.net.xml")

    print(f"\n--- Configuration ---")
    print(f"  Scenario dir  : {scenario_dir}")
    print(f"  Levels        : {', '.join(levels)}")
    print(f"  Method        : {assignment_method}")
    print(f"  Max alternatives: {max_alternatives}")
    print(f"---------------------\n")

    completed = 0
    failed = 0

    for level in levels:
        if run_marouter(scenario_dir, map_file, level, assignment_method, max_alternatives):
            completed += 1
        else:
            failed += 1

    total = len(levels)
    print("=" * 50)
    print("   GENERATION COMPLETE")
    print("=" * 50)
    print(f"✅ Completed: {completed}/{total}")
    print(f"❌ Failed:    {failed}/{total}\n")

    if completed == total:
        print("All wardrop routes generated successfully!")
