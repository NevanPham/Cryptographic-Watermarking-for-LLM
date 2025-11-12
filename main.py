# main.py: Entry point for the CLI

from src.parser import setup_nltk, build_parser
from src.commands import dispatch_command


def main():
    """Main entry point for the CLI."""
    setup_nltk()
    parser = build_parser()
    args = parser.parse_args()
    dispatch_command(args)


if __name__ == "__main__":
    main()
