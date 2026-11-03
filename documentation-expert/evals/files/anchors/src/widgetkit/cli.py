import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="widgetkit")
    ap.add_argument("--dry-run", action="store_true")
    ap.parse_args()
