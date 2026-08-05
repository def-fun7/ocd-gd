import subprocess
import sys
from pathlib import Path

from ocd_gd import (
    HAS_RICH,
    console,
    print_banner,
    print_dataframe_table,
)

# Fallback ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def discover_examples() -> list[Path]:
    """Find all runnable example .py scripts, ignoring runner/helper files."""
    examples_dir = Path(__file__).resolve().parent
    scripts = sorted(examples_dir.glob("*.py"))
    # Exclude runner and internal helpers
    ignored = {"__main__.py"}
    return [s for s in scripts if s.name not in ignored]


def run_script(script_path: Path) -> bool:
    """Execute a single Python script and report success or failure."""
    rel_path = script_path.name

    if HAS_RICH:
        console.print(
            f"\n[bold cyan]▶ Running example:[/bold cyan] [yellow]{rel_path}[/yellow]"
        )
        console.print("[dim]" + "─" * 60 + "[/dim]")
    else:
        print(f"\n{BOLD}▶ Running example:{RESET} {rel_path}")
        print("-" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)], capture_output=False, text=True, check=False
    )

    if HAS_RICH:
        console.print("[dim]" + "─" * 60 + "[/dim]")
        if result.returncode == 0:
            console.print(
                f"[bold green]✔ {rel_path} finished successfully![/bold green]\n"
            )
            return True
        else:
            console.print(
                f"[bold red]❌ {rel_path} exited with error (Code {result.returncode}).[/bold red]\n"
            )
            return False
    else:
        print("-" * 60)
        if result.returncode == 0:
            print(f"{GREEN}✔ {rel_path} finished successfully!{RESET}\n")
            return True
        else:
            print(
                f"{RED}❌ {rel_path} exited with error (Code {result.returncode}).{RESET}\n"
            )
            return False


def main() -> None:
    scripts = discover_examples()

    if not scripts:
        print("No runnable example scripts found.")
        sys.exit(1)

    print_banner("ocd-gd", "Examples Test Suite Runner")

    # Handle running a specific script: python examples 01_single_orbit
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
        matched = [s for s in scripts if target_name in s.name]

        if not matched:
            print(f"Could not find example matching '{target_name}'.")
            print("Available examples:")
            for s in scripts:
                print(f" - {s.name}")
            sys.exit(1)

        success = run_script(matched[0])
        sys.exit(0 if success else 1)

    # Run ALL examples
    passed, failed = 0, 0
    failed_scripts = []

    for script in scripts:
        if run_script(script):
            passed += 1
        else:
            failed += 1
            failed_scripts.append(script.name)

    # Print summary table
    print_dataframe_table(
        title="Execution Summary",
        headers=["Status", "Count"],
        rows=[
            ["Passed", str(passed)],
            ["Failed", str(failed)],
        ],
        header_style="bold magenta",
    )

    if failed_scripts:
        print("\nFailed scripts:")
        for f in failed_scripts:
            print(f" - {f}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
