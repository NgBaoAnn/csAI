"""
Comprehensive test script for seek agent 23120189 against all hide agents with all modes.
Tests various configurations:
- Start modes: deterministic, stochastic
- Observation modes: full visibility, partial visibility
- Pacman speeds: 1, 2, 3
- Capture distances: 1, 2
"""

import subprocess
import sys
import os
from typing import List, Dict, Tuple
from datetime import datetime

os.chdir(os.path.join(os.path.dirname(__file__), "src"))

# All hide agents
HIDE_AGENTS = [
    "hide_bfs",
    "hide_astar",
    "hide_minimax",
    "hide_montecarlo",
    "hide_potential",
]

# Test configurations - each is a dict of arena.py parameters
MODES = [
    {
        "name": "Default (Deterministic, Full Visibility, Speed=2, Capture=2)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "2",
            "pacman-speed": "2",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "Stochastic Start (Random positions)",
        "params": {
            "start-mode": "stochastic",
            "capture-distance": "2",
            "pacman-speed": "2",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "High Speed Pacman (Speed=3)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "2",
            "pacman-speed": "3",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "Low Speed Pacman (Speed=1)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "2",
            "pacman-speed": "1",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "Close Capture Distance (Capture=1)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "1",
            "pacman-speed": "2",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "Far Capture Distance (Capture=3)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "3",
            "pacman-speed": "2",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "Pacman Limited Vision (Obs Radius=3)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "2",
            "pacman-speed": "2",
            "pacman-obs-radius": "3",
            "ghost-obs-radius": "0",
        }
    },
    {
        "name": "Ghost Limited Vision (Obs Radius=3)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "2",
            "pacman-speed": "2",
            "pacman-obs-radius": "0",
            "ghost-obs-radius": "3",
        }
    },
    {
        "name": "Both Limited Vision (Obs Radius=3)",
        "params": {
            "start-mode": "deterministic",
            "capture-distance": "2",
            "pacman-speed": "2",
            "pacman-obs-radius": "3",
            "ghost-obs-radius": "3",
        }
    },
    {
        "name": "Challenging: Stochastic + Limited Vision + Speed1",
        "params": {
            "start-mode": "stochastic",
            "capture-distance": "2",
            "pacman-speed": "1",
            "pacman-obs-radius": "3",
            "ghost-obs-radius": "2",
        }
    },
]

def parse_output(output: str) -> Tuple[str, str]:
    """Parse arena output to extract winner and step count."""
    winner = "UNKNOWN"
    steps = "?"
    
    for line in output.split("\n"):
        if "WINNER" in line:
            if "Ghost" in line:
                winner = "GHOST WINS"
            elif "Pacman" in line:
                winner = "PACMAN WINS"
        if "Total Steps:" in line:
            steps = line.split("Total Steps:")[1].strip()
        if "timed out" in line:
            winner = "TIMEOUT"
        if "Error in" in line:
            winner = "AGENT ERROR"
    
    return winner, steps

def run_test(seek_id: str, hide_id: str, mode_config: Dict) -> Tuple[str, str, str]:
    """Run a single test and return (winner, steps, errors)."""
    cmd = [
        sys.executable, "arena.py",
        "--seek", seek_id,
        "--hide", hide_id,
        "--no-viz",
        "--step-timeout", "3.0",
        "--start-mode", mode_config["params"]["start-mode"],
        "--capture-distance", mode_config["params"]["capture-distance"],
        "--pacman-speed", mode_config["params"]["pacman-speed"],
        "--pacman-obs-radius", mode_config["params"]["pacman-obs-radius"],
        "--ghost-obs-radius", mode_config["params"]["ghost-obs-radius"],
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        winner, steps = parse_output(output)
        errors = ""
        
        if result.returncode != 0 and "AGENT ERROR" in winner:
            err_lines = output.split("\n")
            for line in err_lines:
                if "ERROR" in line:
                    errors = line[:80]
                    break
        
        return winner, steps, errors
    except subprocess.TimeoutExpired:
        return "TIMEOUT (120s)", "N/A", ""
    except Exception as e:
        return "EXCEPTION", "N/A", str(e)[:50]

def main():
    """Main test runner."""
    print("="*100)
    print("COMPREHENSIVE TEST SUITE: Agent 23120189 (Seek) vs All Hide Agents")
    print("="*100)
    print(f"\nTest Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seek Agent: 23120189")
    print(f"Hide Agents: {', '.join(HIDE_AGENTS)}")
    print(f"Test Modes: {len(MODES)}")
    print(f"Total Tests: {len(HIDE_AGENTS) * len(MODES)}")
    print("\n")
    
    # Results storage
    all_results = {}  # mode -> {agent -> (winner, steps, errors)}
    
    # Run tests
    total_tests = len(HIDE_AGENTS) * len(MODES)
    current_test = 0
    
    for mode_idx, mode in enumerate(MODES):
        mode_name = mode["name"]
        print(f"\n{'='*100}")
        print(f"MODE {mode_idx + 1}/{len(MODES)}: {mode_name}")
        print(f"{'='*100}")
        print(f"{'Hide Agent':<20} | {'Result':<20} | {'Steps':<10} | {'Errors':<45}")
        print(f"{'-'*100}")
        
        mode_results = {}
        
        for hide_agent in HIDE_AGENTS:
            current_test += 1
            print(f"\r[{current_test}/{total_tests}] Testing {hide_agent}...", end="", flush=True)
            
            winner, steps, errors = run_test("23120189", hide_agent, mode)
            mode_results[hide_agent] = (winner, steps, errors)
            
            error_str = f"({errors})" if errors else ""
            print(f"\r{hide_agent:<20} | {winner:<20} | {str(steps):<10} | {error_str:<45}")
        
        all_results[mode_name] = mode_results
    
    # Print summary
    print(f"\n\n{'='*100}")
    print("SUMMARY BY MODE")
    print(f"{'='*100}\n")
    
    for mode_idx, mode_name in enumerate([m["name"] for m in MODES], 1):
        results = all_results[mode_name]
        pacman_wins = sum(1 for w, _, _ in results.values() if w == "PACMAN WINS")
        ghost_wins = sum(1 for w, _, _ in results.values() if w == "GHOST WINS")
        errors = sum(1 for w, _, _ in results.values() if "ERROR" in w or "TIMEOUT" in w)
        
        print(f"Mode {mode_idx}: {mode_name}")
        print(f"  Pacman Wins: {pacman_wins}/{len(HIDE_AGENTS)} ✓")
        print(f"  Ghost Wins:  {ghost_wins}/{len(HIDE_AGENTS)} ✗")
        print(f"  Errors:      {errors}/{len(HIDE_AGENTS)}")
        print()
    
    # Print summary by agent
    print(f"\n{'='*100}")
    print("SUMMARY BY HIDE AGENT")
    print(f"{'='*100}\n")
    
    for hide_agent in HIDE_AGENTS:
        pacman_wins = 0
        ghost_wins = 0
        errors = 0
        
        for mode_name in all_results:
            w, _, e = all_results[mode_name].get(hide_agent, ("UNKNOWN", "?", ""))
            if w == "PACMAN WINS":
                pacman_wins += 1
            elif w == "GHOST WINS":
                ghost_wins += 1
            else:
                errors += 1
        
        print(f"{hide_agent:<20}: Pacman {pacman_wins}/{len(MODES)} | Ghost {ghost_wins}/{len(MODES)} | Errors {errors}/{len(MODES)}")
    
    # Overall statistics
    print(f"\n{'='*100}")
    print("OVERALL STATISTICS")
    print(f"{'='*100}\n")
    
    total_pacman_wins = 0
    total_ghost_wins = 0
    total_errors = 0
    
    for mode_name in all_results:
        for w, _, e in all_results[mode_name].values():
            if w == "PACMAN WINS":
                total_pacman_wins += 1
            elif w == "GHOST WINS":
                total_ghost_wins += 1
            else:
                total_errors += 1
    
    total = total_pacman_wins + total_ghost_wins + total_errors
    pacman_percentage = (total_pacman_wins / total * 100) if total > 0 else 0
    
    print(f"Total Tests Run:     {total}")
    print(f"Pacman Wins:         {total_pacman_wins} ({pacman_percentage:.1f}%)")
    print(f"Ghost Wins:          {total_ghost_wins} ({100-pacman_percentage:.1f}%)")
    print(f"Errors/Timeouts:     {total_errors}")
    print(f"\n✓ Seek Agent Win Rate: {pacman_percentage:.1f}%")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
