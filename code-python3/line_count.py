# line_count.py
import sys

if __name__ == "__main__":

    count = 0
    for line in sys.stdin:
        count += 1

    # print goes to sys.stdout
    print(count)
