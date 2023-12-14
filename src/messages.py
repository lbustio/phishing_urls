def welcome_message():
    print("#> Script que crea el dataset de entrenamiento a partir de archivo de entrada con las URLS.")
    print("#> La estructura del archivo de entrada es: url, label.")
    print("#> Este script está basado en el paper 'Detecting Malicious URLs Using Lexical Analysys' y ")
    print("#> 'Intelligent phishing url using association rules'.")
    print("#> 2019.12.05 - Lazaro Bustio Martinez (lbustio@inaoep.mx)")


def print_resume(dataset):
    rows, cols = dataset.shape
    print("#>    Number of rows: %s" % rows)
    print("#>    Number of columns: %s" % cols)
    features = " ".join(dataset[0, :])
    print("#>    Features: %s" % features)
