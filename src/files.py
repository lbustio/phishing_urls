import csv
import numpy as np
import pickle


# read the urls input file
def read_url_file(filePath):
    urls = []
    labels = []

    # reading csv file
    with open(filePath, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        url = ""
        label = ""
        features = next(csvreader)
        features = [x for x in features if x]
        for row in csvreader:
            r = [x for x in row if x]
            urls.append(r[0])
            labels.append(r[1])


    result = np.column_stack((urls, labels))
    print(result.shape)

    # get total number of rows
    num_rows = csvreader.line_num - 1
    print("#>   The input file was read and a dataset containing %s rows and %s columns was created." % (
        result.shape[0], result.shape[1]))

    return result


# saves the features dataset as a CSV file.
def write_features_file(filePath, dataset):
    with open(filePath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)
        print("#>   File saved!!!")


# read the features input file
def read_features_file(filePath):

    values = []
    # reading csv file
    with open(filePath, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        features = next(csvreader)
        values.append(features)
        for row in csvreader:
            ini_array = np.array(row)
            # conerting to array of floats
            # using np.astype
            res = ini_array.astype(np.float)
            values.append(res)

    result = np.array(values)

    # get total number of rows
    num_rows = csvreader.line_num - 1
    print("#>   The input file was read and a dataset containing %s rows and %s columns was created." % (
        result.shape[0], result.shape[1]))

    return result


# Saves the training model using the provided filename
def write_training_model(model, filename):
    # save the model to disk
    pickle.dump(model, open(filename, "wb"))


# read the training model
def read_training_model(filename):
    # load the model from disk
    mdl = pickle.load(open(filename, "rb"))
    return mdl


# read the mutual information file precalculated
def read_mutual_information_file(filePath):
    with open(filePath, 'r') as csvfile:
        # getting mutual information
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        n = []
        p = []

        next(csvreader)
        for row in csvreader:
            r = [x for x in row if x]
            line = []
            line.append(r[1])  # Term
            line.append(r[2])  # mutual information
            line.append(r[3])  # repetitions
            line.append(r[4])  # value
            if (r[0] == "0"):
                n.append(line)
            else:
                p.append(line)

        phishing = np.array(p)
        normal = np.array(n)

    return normal, phishing


# read a csv file
def read_csv_file(filePath):

    values = []
    # reading csv file
    with open(filePath, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        features = next(csvreader)
        values.append(features)
        for row in csvreader:
            values.append(row)

    result = np.array(values)

    return result


# saves the features dataset as a CSV file.
def write_csv_file(filePath, dataset):
    with open(filePath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)
        print("#>   File saved!!!")
