#!/usr/bin/env python3
"""Quick status check for the dedup pipeline."""
import subprocess
import json

def get_logs():
    """Get container logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "dedup-dedup-1"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"

def get_db_status():
    """Check database status."""
    try:
        result = subprocess.run(
            [
                "docker", "exec", "dedup-postgres",
                "psql", "-U", "dedup", "-d", "dedup", "-c",
                "SELECT COUNT(*) as source_files, COUNT(DISTINCT sha256) as unique_hashes FROM source_files; SELECT COUNT(*) as unique_files FROM unique_files;"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 60)
    print("DEDUP PIPELINE STATUS CHECK")
    print("=" * 60)

    # Last few log lines
    logs = get_logs()
    lines = logs.split('\n')
    print("\nRecent logs:")
    for line in lines[-10:]:
        if line.strip():
            print(f"  {line}")
    # Database status
    print("\nDatabase status:")
    db_status = get_db_status()
    print(db_status)

if __name__ == "__main__":
    main()
