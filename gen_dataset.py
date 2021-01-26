import argparse
from src.data.generate_dataset import dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--category', default='people', help='People, penguins or vehicles')
    parser.add_argument('--dataset-name', default='people.3.0.0', help='Dataset identifier')
    parser.add_argument('--variations', default=3, help='Dataset size')
    parser.add_argument('--max', default=3, help='Dataset size')
    parser.add_argument('--workers', default=1, help='Number of workers')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    dataset(args)

    
