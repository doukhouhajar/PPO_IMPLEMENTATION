import argparse
import os
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='name of the experiement')
    parser.add_argument('--gym-id', type=str, default="CartPol-v1",
                        help='gym env id')