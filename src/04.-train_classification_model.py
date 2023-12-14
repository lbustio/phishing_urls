import src

from src.files import read_url_file
from src.messages import print_resume
from src.messages import welcome_message
from src.classify import classify
from src.classify import classify_individual
from src.classify import classify_cummulative
from src.files import write_training_model



def main():
    inputWin = r"C:\Users\lbust\Documents\PyCharmProjects\phishing\data\FeaturesDatasetReduced.csv"
    outputWin = r"C:\Users\lbust\Documents\PyCharmProjects\phishing\data\TrainModel.mdl"
    inputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/FeaturesDataset.csv"
    outputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/FTrainModel.mdl"
    input = inputWin
    output = outputWin

    welcome_message()

    print("#> Reading file from: %s" % input)
    dataset = src.files.read_features_file(input)
    print("#> Dataset correctly read from: %s" % input)
    print("#> Dataset resume:")
    print_resume(dataset)

    rows, cols = dataset.shape
    urls = dataset[:, 0 : cols - 1]
    labels = dataset[:, -1]
    k_fold = 1
    mdl = 0
    mdl = classify(urls, labels, k_fold)
    #classify_individual(urls, labels, k_fold)
    #classify_cummulative(urls, labels, k_fold)

    if k_fold == 1:
        print("#> Saving the training model as 'train_model.mdl'")
        write_training_model(mdl, output)

    print("#> Training process completed!!!!")
    exit(0)


if __name__ == "__main__":
    main()
