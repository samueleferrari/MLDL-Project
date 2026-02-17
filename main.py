import argparse
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--d", type=str, default=None, help="Path to the dataset")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')

if __name__ == '__main__':
    main() 
