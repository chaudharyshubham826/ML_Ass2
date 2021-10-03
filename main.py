from Q1.Q1 import NaiveBayes

# import warnings
# warnings.filterwarnings("ignore")


print("Enter the question number: ")
q = int(input())

if(q == 1):
    nb = NaiveBayes("Music_Review_train.json", "Music_Review_test.json")
    input("Press Enter to run Part A...")
    nb.part1a()
    input("Press Enter to run Part B...")
    nb.part1b()

    input("Press Enter to run Part C...")
    nb.part1c()
