import os
x = 2
highest_number = 0
directory = '/home/sinem/Schreibtisch/Text/'

def change_max(filename, x_coord, highest_number):
    with open("Text/"+filename, "r") as input:
        with open("fixed/"+filename, "a") as output:
            for line in input:
                if line.split(' ')[1] == str(x_coord):
                    if int(line.split(' ')[2])>highest_number:
                        highest_number = int(line.split(' ')[2])
                        end = str(highest_number) + ' '+ line.split(' ')[3] + ' ' + line.split(' ')[4]
            output.write("test_images/Datensatz/"+str(x)+'.png' + " " + str(x_coord) + " " + end)




for filename in os.listdir(directory):
    iteration = 0
    loop = 0
    while loop <= 239:
        change_max(str(x)+'.txt', iteration, highest_number)
        iteration+=8
        highest_number=0
        loop+=1
    x+=1

