import numpy as np
import time

from src.messages import welcome_message
from src.files import read_training_model
from src.features import path_url_rate_calculator
from src.features import hostname_url_rate_calculator
from src.features import letter_hostname_rate_calculator
from src.features import get_mutual_info_average
from src.features import domain_url_rate_calculator
from src.features import slash_counter
from src.features import path_entropy_calculator
from src.features import url_entropy_calculator
from src.files import read_mutual_information_file


def main():
    # inputWin = r"C:\Users\lbust\Documents\PyCharmProjects\phishing\data\TrainModel.mdl"
    # inputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/TrainModel.mdl"
    # input = inputWin
    input = "./data/TrainModel.mdl"

    welcome_message()

    print("#> Reading model from: %s" % input)
    mdl = read_training_model(input)
    print("#> Training model read!!!")
    print("#> Preparing unknown data for prediction...")
    urls = ["https://www.inaoep.mx", "http://03418f6.netsolhost.com/FF7AADF203DF6C7A0B7C8A74B8164E55/?sec=Puc Kotsis"]
    urls = np.array(urls)
    data = []
    # phishing, normal = read_mutual_information_file(
    #     r"C:\Users\lbust\Documents\PyCharmProjects\phishing\data\Mutual_Info_Ordered.csv")
    phishing, normal = read_mutual_information_file('./data/Mutual_Info_Ordered.csv')

    for url in urls:
        # F37: calculate the path - url rate
        path_rate = path_url_rate_calculator(url)

        # F36: calculate the hostname - url rate
        hostname_rate = hostname_url_rate_calculator(url)

        # F44: Letter-hostname rate
        letter_hostname_rate = letter_hostname_rate_calculator(url)

        # F32 and F33: get mutual information average for phishing and et mutual information
        # average for legitimate
        mi_phishing, mi_normal = get_mutual_info_average(url, phishing, normal)

        # F34: calculate the domain - url rate
        domain_rate = domain_url_rate_calculator(url)

        # F14: Number of slashes in the url.
        slash_count = slash_counter(url)

        # F31: calculate the entropy of the path
        path_entropy = path_entropy_calculator(url)

        # F29: calculate the entropy of the url
        url_entropy = url_entropy_calculator(url)

        line = [slash_count, url_entropy, path_entropy, mi_phishing, mi_normal, domain_rate,
                hostname_rate, path_rate, letter_hostname_rate]

        data.append(line)

    print("#> Prepared data!")
    print("#> predicting unknown data...")
    start_time = time.time()
    label = mdl.predict(data)
    print("#>     Time needed: %s seconds" % (time.time() - start_time))
    print("#> Predicted class for the unknown data: %s" % label)


if __name__ == "__main__":
    main()
