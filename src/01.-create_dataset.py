import src
from src.features import get_features
from src.files import read_url_file
from src.files import write_features_file
from src.messages import print_resume
from src.messages import welcome_message
from info_gain import info_gain


def main():
    inputWin = r"C:\Users\lbustio\Documents\PycharmProjects\phishing\phishing\data\URLs.csv"
    outputWin = r"C:\Users\lbustio\Documents\PycharmProjects\phishing\phishing\data\FeaturesDataset.csv"
    inputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/URLs-26k.csv"
    outputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/FeaturesDataset.csv"
    input = inputMac
    output = outputMac

    welcome_message()

    print("#> Reading file from: %s" % input)
    urls = src.files.read_url_file(input)
    dataset = get_features(urls)
    print("#> Dataset correctly created from the input file!")
    print("#> Dataset resume:")
    print_resume(dataset)
    print("#> Saving dataset in the output %s" % output)
    write_features_file(output, dataset)
    print("#> Dataset correctly saved in the output!!!")
    print("#> Dataset creation done!!!!")


if __name__ == "__main__":
    main()
