"""Minimal CLI interface for Atuna."""

import argparse
import sys

from .registry import model_registry
from . import __version__


def main() -> None:
    """CLI entry point for Atuna."""
    parser = argparse.ArgumentParser(
        description="Atuna: Fine-tuning assistant for large language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atuna --version                   Show version information
  atuna --list-models              List available models

For advanced usage, use Atuna as a Python library:
  from atuna import Tuna, TunaConfig, TrainingConfig, model_registry
        """,
    )

    parser.add_argument("--version", action="version", version=f"Atuna {__version__}")

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models in the registry",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model_name, config in model_registry.items():
            print(f"  {model_name}")
            print(f"    Chat template: {config.chat_template}")
            print(f"    Temperature: {config.temperature}")
            print(f"    Top-p: {config.top_p}")
            print(f"    Top-k: {config.top_k}")
            print()
        return

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nFor full functionality, use Atuna as a Python library.")
        print("See examples/ directory for usage examples.")


if __name__ == "__main__":
    main()
