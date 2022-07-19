import os
x = 2
directory = '/home/sinem/Schreibtisch/Textdatei/'

def delete(filename):
    with open("Textdatei/"+filename, "r") as input:
        with open("Text/"+filename, "w") as output:
            # iterate all lines from file
            for line in input:
                # if substring contain in a line then don't write it
                if (int(line.split(' ')[4])!=10):
                    output.write(line)

for filename in os.listdir(directory):
    delete(str(x)+'.txt')
    x+=1
'''
def stixelNet(filename, x_coord, y_coord):
    with open(filename) as f:
        for x_coord in f where y_coord
        s = f.read()
'''