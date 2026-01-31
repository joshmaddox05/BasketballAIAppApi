#!/usr/bin/env python3
"""
Convenient test runner for Basketball Shot Analysis API

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Run quick tests only (skip slow integration)
    python run_tests.py --video      # Run only video analysis tests
    python run_tests.py --summary    # Run summary report only
    python run_tests.py --landmarks  # Run landmark tests only
"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Basketball API tests')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (skip video analysis)')
    parser.add_argument('--video', action='store_true',
                       help='Run video analysis tests only')
    parser.add_argument('--summary', action='store_true',
                       help='Run summary report only')
    parser.add_argument('--landmarks', action='store_true',
                       help='Run landmark configuration tests only')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage report')

    args = parser.parse_args()

    cmd = ['python', '-m', 'pytest']

    if args.quick:
        cmd.extend(['tests/test_landmarks.py', '-v'])
    elif args.video:
        cmd.extend(['tests/test_video_analysis.py', '-v', '-s'])
    elif args.summary:
        cmd.extend(['tests/test_video_analysis.py::TestSummaryReport', '-v', '-s'])
    elif args.landmarks:
        cmd.extend(['tests/test_landmarks.py', '-v'])
    else:
        cmd.extend(['-v', '-s'])

    if args.coverage:
        cmd = ['python', '-m', 'pytest', '--cov=services', '--cov-report=term-missing'] + cmd[3:]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
