def print_pattern(rows=10):
    for i in range(rows):
        for j in range(i):
            print(i, end=' ')
        print('')

if __name__ == "__main__":
    print_pattern()