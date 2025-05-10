lists = [
    list(range(1, 51)),
    list(range(1, 12)),
    list(range(1, 16)),
    list(range(1, 12)),
    list(range(1, 27)),
    list(range(1, 24))
]

global_positions = []
offset = 0

for lst in lists:
    positions = list(range(offset + 1, offset + len(lst) + 1))
    global_positions.append(positions)
    offset += len(lst)

# Display results
for i, pos in enumerate(global_positions):
    print(f"List {i+1} global positions: {pos}")





a = 1
b = 2

c = a + b 

b = 1

c = a + b 