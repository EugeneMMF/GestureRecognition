result = []
with open("coordinates_data.txt", "r") as f:
    for line in f:
        res = line.split(",")
        arr = ",".join(i for i in res[:-1])
        result.append(arr)

with open("coordinates_data.txt", "w") as f:
    for line in result:
        f.write(line)
        f.write("\n")