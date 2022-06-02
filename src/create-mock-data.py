values = 300

counter100 = 0
counter200 = 0
counter300 = 0
counter400 = 0
counter500 = 0
counter600 = 0
counter700 = 0
counter800 = 0
output_string = ""

step_amount = 5

for i in range(values):
    if counter100 < step_amount:
        value = "100"
        counter100 += 1 
    elif counter200 < step_amount:
        value = "200"
        counter200 += 1
    elif counter300 < step_amount:
        value = "300"
        counter300 += 1
    elif counter400 < step_amount:
        value = "400"
        counter400 += 1
    elif counter500 < step_amount:
        value = "500"
        counter500 += 1
    elif counter600 < step_amount:
        value = "600"
        counter600 += 1
    elif counter700 < step_amount:
        value = "700"
        counter700 += 1
    elif counter800 < step_amount:
        value = "800"
        counter800 += 1
    else:
        counter100 = 0
        counter200 = 0
        counter300 = 0
        counter400 = 0
        counter500 = 0
        counter600 = 0
        counter700 = 0
        counter800 = 0
    output_string += f'{value} '

with open("step-data.txt", "w") as f:
    f.write(output_string)