from Q1.Q1 import NaiveBayes
import sys

# import warnings
# warnings.filterwarnings("ignore")



nb = NaiveBayes(sys.argv[1], sys.argv[2])
input("Press Enter to run Part A...")
nb.part1a()
input("Press Enter to run Part B...")
nb.part1b()

input("Press Enter to run Part C...")
nb.part1c()

input("Press Enter to run Part D...")
nb.part1d()

input("Press Enter to run Part E(Feature 1)...")
nb.feature1()

input("Press Enter to run Part E(Feature 2)...")
nb.feature2()

input("Press Enter to run Part F...")
nb.part1f()

input("Press Enter to run Part G...")
nb.part1g()
