import os
import sys
import shutil
import subprocess
from pathlib import Path


def _print_kv(key: str, value: str | None) -> None:
    print(f"{key}: {value if value else '(not set)'}")


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _try_imports() -> tuple[bool, str]:
    try:
        import traci  # noqa: F401
        import sumolib  # noqa: F401

        return True, "Imported traci + sumolib"
    except Exception as exc:  # noqa: BLE001
        return False, f"Import failed: {exc!r}"


def _ensure_sumo_tools_on_path() -> bool:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        return False

    tools_dir = Path(sumo_home) / "tools"
    if not tools_dir.exists():
        return False

    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    return True


def main() -> int:
    print("== Mono_QMIX Step 1 setup check ==")
    _print_kv("python", sys.version.split("\n")[0])
    _print_kv("executable", sys.executable)

    sumo_home = os.environ.get("SUMO_HOME")
    _print_kv("SUMO_HOME", sumo_home)

    sumo = _which("sumo")
    netgenerate = _which("netgenerate")
    sumo_gui = _which("sumo-gui")

    _print_kv("which(sumo)", sumo)
    _print_kv("which(netgenerate)", netgenerate)
    _print_kv("which(sumo-gui)", sumo_gui)

    if sumo:
        try:
            out = subprocess.check_output([sumo, "--version"], text=True, stderr=subprocess.STDOUT)
            print("-- sumo --version --")
            print(out.strip())
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to run sumo --version: {exc!r}")

    ok, msg = _try_imports()
    if not ok:
        added = _ensure_sumo_tools_on_path()
        if added:
            ok2, msg2 = _try_imports()
            ok, msg = ok2, msg2 + " (after adding SUMO_HOME/tools to sys.path)"

    print("-- python imports --")
    print(msg)

    if not sumo:
        print("\nERROR: SUMO binary not found on PATH.")
        print("- Install SUMO and add its 'bin' directory to PATH.")
        return 2

    if not os.environ.get("SUMO_HOME"):
        print("\nWARNING: SUMO_HOME is not set.")
        print("- Many tools expect SUMO_HOME to point to the SUMO install root.")

    if not ok:
        print("\nERROR: Could not import traci/sumolib.")
        print("- Either `pip install -r requirements.txt` OR ensure SUMO_HOME/tools is on PYTHONPATH.")
        return 3

    print("\nOK: Tooling looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
