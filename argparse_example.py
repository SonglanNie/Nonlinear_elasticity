import argparse
from firedrake import *
# import numpy
# from tabulate import tabulate
 
parser = argparse.ArgumentParser(description='A simple example of argparse usage.')
parser.add_argument('-n', '--name', type=str, help='Your name')
parser.add_argument('-a', '--age', type=int, help='Your age')
 
args = parser.parse_args()
# args, _ = parser.parse_known_args()

print("Hello,",args.name,"! You are",args.age,"years old.")

age = Constant(args.age)
print(age)