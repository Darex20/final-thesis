values = 1000

counter100 = 0
counter200 = 0
output_string = ""

for i in range(values):
    if counter100 < 3:
        value = "100"
        output_string += f'{value} '
        counter100 += 1 
    elif counter200 < 3:
        value = "200"
        output_string += f'{value} '
        counter200 += 1
    else:
        counter100 = 0
        counter200 = 0

with open("step-data.txt", "w") as f:
    f.write(output_string)