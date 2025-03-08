import time

for i in range(1, 101):
    print(f"{i} \t tab{i/100}", end='', flush = True)
    time.sleep(0.1)  # Optional delay to see the numbers change