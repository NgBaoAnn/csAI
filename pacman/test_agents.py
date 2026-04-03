"""Quick test script for all 5 Ghost agents."""
import subprocess
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "src"))

agents = ["hide_bfs", "hide_astar", "hide_minimax", "hide_montecarlo", "hide_potential"]

results = []
for agent in agents:
    cmd = [
        sys.executable, "arena.py",
        "--seek", "23120189",
        "--hide", agent,
        "--no-viz",
        "--step-timeout", "3.0"
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = out.stdout + out.stderr
        
        winner = "UNKNOWN"
        steps = "?"
        
        for line in output.split("\n"):
            if "WINNER" in line:
                if "Ghost" in line:
                    winner = "GHOST WINS ✅"
                elif "Pacman" in line:
                    winner = "PACMAN WINS ❌"
            if "Total Steps:" in line:
                steps = line.split("Total Steps:")[1].strip()
            if "timed out" in line:
                winner = "TIMEOUT (ghost default win)"
            if "Error in" in line:
                winner = "AGENT ERROR"
        
        results.append((agent, winner, steps))
        print(f"{agent:20s} | {winner:30s} | Steps: {steps}")
        
        if out.returncode != 0 and winner == "UNKNOWN":
            err_lines = out.stderr.strip().split("\n")
            for el in err_lines[:3]:
                print(f"  ERR: {el}")
                
    except subprocess.TimeoutExpired:
        results.append((agent, "TIMEOUT (120s)", "N/A"))
        print(f"{agent:20s} | TIMEOUT (120s)")
    except Exception as e:
        results.append((agent, f"ERROR: {e}", "N/A"))
        print(f"{agent:20s} | ERROR: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for agent, result, steps in results:
    print(f"  {agent:20s} | {result:30s} | Steps: {steps}")
