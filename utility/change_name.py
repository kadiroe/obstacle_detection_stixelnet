import os
x = 9527
x_old = 9
directory = '/home/sinem/Dokumente/Datensatz/Textdatei/'

def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)

for filename in os.listdir(directory):
	y = str(x)
	y_old = f'{x_old:06}'
	inplace_change(y+'.txt','test_images/Datensatz/'+y_old+'.png', 'test_images/Datensatz/'+y+'.png')
	x+=1
	x_old+=40