import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n', type=int)
    
    args = parser.parse_args()


print(f'Program {args.n} Ran!')