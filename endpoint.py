from flask import Flask, request, make_response
import datetime
from flask_cors import CORS
import urllib.parse
from src.files import read_training_model,read_url_file
from src.features import path_url_rate_calculator
from src.features import hostname_url_rate_calculator
from src.features import letter_hostname_rate_calculator
from src.features import get_mutual_info_average
from src.features import domain_url_rate_calculator
from src.features import slash_counter
from src.features import path_entropy_calculator
from src.features import url_entropy_calculator
from src.files import read_mutual_information_file

app = Flask(__name__)
CORS(app)

model = None
phishing = None
normal = None


def load_model():
    global model, phishing, normal
    dir_model = "./data/TrainModel.mdl"
    model = read_training_model(dir_model)
    print("#> Training model read!!!")
    phishing, normal = read_mutual_information_file(
        "./data/Mutual_Info_Ordered.csv")


@app.route('/url/hello', methods=['GET'])
def hello():
    return datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 200


@app.route('/url/validate', methods=['GET'])
def validate():
    global model,phishing,normal
    url = request.args.get('url', '')
    urllib.parse.quote_plus(url)
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
    label = model.predict([line])
    to_print = 'legítimo'
    if label[0] == '0.0':
        to_print = 'phishing'
    return '{} --> {}({})'.format(url,label[0],to_print), 200



if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5001)
