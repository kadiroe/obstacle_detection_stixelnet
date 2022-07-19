import os


def main():
    path = "/home/sinem/Schreibtisch/Datensatz/Textdatei/"
    image_list = [f for f in os.listdir(os.path.join(os.getcwd(), path)) if f.endswith('.txt')]

    files = len(image_list)
    training = files*0.8
    test = training+files*0.1
    validate = files
    data = []
    data_test = []
    data_validate = []
    for index, file_name in enumerate(image_list):
        if (index <= training):
            with open(path + "/" + file_name) as fp:
                data.append(fp.read())
        elif (training < index <= test):
            with open(path + "/" + file_name) as fp1:
                data_test.append(fp1.read())
        elif (test < index <= validate):
            with open(path + "/" + file_name) as fp2:
                data_validate.append(fp2.read())

    with open("training_data.txt", 'w') as fp:
        fp.write(''.join(data))
    with open("test_data.txt", 'w') as fp1:
        fp1.write(''.join(data_test))
    with open("validation_data.txt", 'w') as fp2:
        fp2.write(''.join(data_validate))


if __name__ == "__main__":
    main()

