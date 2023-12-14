import src
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.files import read_url_file
from src.messages import welcome_message
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from src.mutual_information import informacionMutua
from info_gain import info_gain



def numpy2pandas(input_numpy):
    r,c = input_numpy.shape
    rows = [i for i in range(r-1)]
    data = input_numpy[1:, 0:].astype(float)
    columns = input_numpy[0, 0:]
    result = pd.DataFrame(data=data, index = rows, columns = columns)
    return result


# Prints some statistics of the dataset
def print_stats(dataset):
    stats = dataset.describe()
    print(stats)
    print(dataset.groupby("labels").size())

    return stats


# Show some tendencies for the dataset
def dispersion(dataset):
    dataset.drop(["labels"], 1).hist()
    plt.show()


# Histogram plotting
def histogram(dataset, label, color, bins, title, xlabel):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})
    sns.distplot(dataset, kde=False, bins=bins, color=color, label=label);
    plt.title(title % label)
    plt.legend(loc="upper right")
    #plt.xlabel(xlabel % bins)
    plt.ylabel("Frequency")
    plt.show()


# URL Length analysis
def url_length_distribution(dataset):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})

    data1 = dataset[dataset["labels"] == 0]["url_len"]
    sns.distplot(data1, color="green", label="Legitimous URLs")
    data2 = dataset[dataset["labels"] == 1]["url_len"]
    sns.distplot(data2, color="red", label="Phishing URLs")
    plt.title("URL Length Distribution")
    plt.legend(loc="upper right")
    plt.xlabel("Length of URL")
    plt.ylabel("Probability density")
    plt.show()

    bins1 = data1.max()
    histogram(data1, "Legitimous URLs", "green", bins1, "URL Length Histogram for class %s", "Length of URL (max %s)")
    bins2 = data2.max()
    histogram(data2, "Phishing URLs", "red", bins2, "URL Length Histogram for class %s", "Length of URL (max %s)")


# Host length analysis
def host_length_distribution(dataset):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})
    data1 = dataset[dataset["label"] == 0]["host len"]
    sns.distplot(data1, color="green", label="Benign URLs")
    data2 = dataset[dataset["label"] == 1]["host len"]
    sns.distplot(data2, color="red", label="Phishing URLs")
    plt.title("Host Length Distribution")
    plt.legend(loc="upper right")
    plt.xlabel("Length of hostname")
    plt.ylabel("Probability density")
    plt.show()

    bins1 = data1.max()
    histogram(data1, "Legitimous URLs", "green", bins1, "Host Length Histogram for class %s", "Length of Host (max %s)")
    bins2 = data2.max()
    histogram(data2, "Phishing URLs", "red", bins2, "Host Length Histogram for class %s", "Length of URL (max %s)")


# Analysis of the number of dots in host
def host_dots_distribution(dataset):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})
    data1 = dataset[dataset["label"] == 0]["host dots"]
    sns.distplot(data1, color="green", label="Benign URLs")
    data2 = dataset[dataset["label"] == 1]["host dots"]
    sns.distplot(data2, color="red", label="Phishing URLs")
    plt.title("Host Dots Distribution")
    plt.legend(loc="upper right")
    plt.xlabel("Number of host dots")
    plt.ylabel("Probability density")
    plt.show()

    bins1 = data1.max()
    histogram(data1, "Legitimous URLs", "green", bins1, "Host dots Histogram for class %s",
              "Number of dots in host (max %s)")
    bins2 = data2.max()
    histogram(data2, "Phishing URLs", "red", bins2, "Host dots Histogram for class %s",
              "Number of dots in dots (max %s)")


# Analysis of the number of dots in the url
def url_dots_distribution(dataset):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})
    data1 = dataset[dataset["label"] == 0]["url dots"]
    sns.distplot(data1, color="green", label="Benign URLs")
    data2 = dataset[dataset["label"] == 1]["url dots"]
    sns.distplot(data2, color="red", label="Phishing URLs")
    plt.title("URL Dots Distribution")
    plt.legend(loc="upper right")
    plt.xlabel("Number of URL dots")
    plt.ylabel("Probability density")
    plt.show()

    bins1 = data1.max()
    histogram(data1, "Legitimous URLs", "green", bins1, "URL dots Histogram for class %s",
              "Number of dots in host (max %s)")
    bins2 = data2.max()
    histogram(data2, "Phishing URLs", "red", bins2, "URL dots Histogram for class %s",
              "Number of dots in dots (max %s)")


# Analyzin the slashes distribution in the url
def slashes_distribution(dataset):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})
    data1 = dataset[dataset["label"] == 0]["slashes"]
    sns.distplot(data1, color="green", label="Benign URLs")
    data2 = dataset[dataset["label"] == 1]["slashes"]
    sns.distplot(data2, color="red", label="Phishing URLs")
    plt.title("Slashes Distribution")
    plt.legend(loc="upper right")
    plt.xlabel("Number of URL slashes")
    plt.ylabel("Probability density")
    plt.show()

    bins1 = data1.max()
    histogram(data1, "Legitimous URLs", "green", bins1, "Slashes Histogram for class %s",
              "Number of slashes in url (max %s)")
    bins2 = data2.max()
    histogram(data2, "Phishing URLs", "red", bins2, "Slashes Histogram for class %s",
              "Number of slashes in url (max %s)")


# Clusters analysis
def cluster_analysis(dataset):
    x = np.array(dataset.drop(["label", "url"], axis=1))
    y = np.array(dataset["label"])

    nc = range(1, 24)
    kmeans = [KMeans(n_clusters=i) for i in nc]
    score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
    plt.plot(nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()

    clusters_number = 5  # this value is obtained from the Elbow curve
    kmeans = KMeans(n_clusters=clusters_number).fit(x)
    centroids = kmeans.cluster_centers_
    print(centroids)

    # Predicting the clusters
    labels = kmeans.predict(x)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    colores = ["red", "green", "blue", "cyan", "magenta"]
    # colores = ["red", "green"]
    asignar = []
    for row in labels:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=asignar, s=60)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
    plt.show()

    """
    # Getting the values and plotting it
    features_name = ["url", "host len", "slashes", "host dots", "terms", "special chars",
                     "contains ip", "unicode chars", "is https", "subdomains", "keywords",
                     "toplevel", "path dots", "host hyphen", "url len", "is tiny", "num at",
                     "has redirect", "count delimiters", "num subdirs", "count query",
                     "url entropy", "host entropy", "path entropy", "url dots", "label"]

    for i in features_name:
        f1 = dataset[i].values
        x_label = i
        for j in features_name:
            f2 = dataset[j].values
            plt.scatter(f1, f2, c=asignar, s=70)
            plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
            plt.xlabel(x_label)
            y_label = j
            plt.ylabel(y_label)
            plt.legend(loc="upper right")
            plt.title(x_label + " vs " + y_label)
            plt.show()
    """


# PCA analysis
def pca_plotting(dataset):
    x = np.array(dataset.drop(["url", "label"], axis=1))
    y = np.array(dataset["label"])

    pca = sklearnPCA(n_components=2)
    x_norm = (x - x.min()) / (x.max() - x.min())
    transformed = pd.DataFrame(pca.fit_transform(x_norm))
    plt.scatter(transformed[y == 0][0], transformed[y == 0][1], label="Legitimous URLs", c="green")
    plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label="Phishing URLs", c="red")
    plt.legend(loc="upper right")
    plt.title("PCA plot of the input dataset")
    plt.show()



def InfoGain(dataset):
    ig = {}
    label = dataset[1:,-1]
    dataset = dataset[:,:-1]
    for column in dataset.T:
        feature = column[0]
        values = column[1:]
        ig[feature] = info_gain.info_gain_ratio(values, label)

    return np.array(ig)


def main():
    inputWin = r""
    inputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/FeaturesDataset-26k.csv"
    input = inputMac

    print("#> Script para obtener algunos análisis exploratorios del feature dataset.")
    print("#> Reading file from: %s" % input)
    features = src.files.read_features_file(input)
    print("#> Read the input file!")

    ig = InfoGain(features)


    feat = numpy2pandas(features)

    plt.title("Test")
    data1 = feat[feat["labels"] == 0]["url_len"]
    b = data1.nunique()
    plt.hist(data1, bins=b, alpha=1, edgecolor='black', linewidth=1)
    plt.grid(True)
    plt.show()

    data2 = feat[feat["labels"] == 1]["url_len"]
    b = data2.nunique()
    plt.hist(data2, bins=b, alpha=1, edgecolor='black', linewidth=1)
    plt.grid(True)
    plt.show()

    data3 = feat["url_len"]
    b = data3.nunique()
    plt.hist(data3, bins=b, alpha=1, edgecolor='black', linewidth=1)
    plt.grid(True)
    plt.show()

    plt.clf()

    url_length_distribution(feat)

    # obtaining mutual information
    #informacionMutia(input)
    #print("termine")

    # obtaining a general view of the dataset
    #dispersion(feat)

    # obtaining the distribution of the lengths of the urls
    #url_length_distribution(feat)

    # obtaining the distribution of the lengths of the host
    #host_length_distribution(urls)

    # obtaining the distribution of the dots in the url
    #url_dots_distribution(urls)

    # Obtaining the distribution of the slashes in the url
    #slashes_distribution(urls)

    # generating clusters from the dataset
    #cluster_analysis(urls)

    # Representing the dataset using 2 or 3 principal components
    #pca_plotting(urls)



if __name__ == "__main__":
    main()