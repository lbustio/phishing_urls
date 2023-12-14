import src
from src.files import read_csv_file
from src.files import write_csv_file

import statistics
import numpy as np

def main():
    # Values for IG, CS and RfF were calculated outside this script using Weka. Results of
    # weka were used to create a csv file which is used by this script.
    igWin = r"C:\Users\lbustio\Documents\PycharmProjects\phishing\phishing\data\URLs.csv"
    csWin = r"C:\Users\lbustio\Documents\PycharmProjects\phishing\phishing\data\URLs.csv"
    rffWin = r"C:\Users\lbustio\Documents\PycharmProjects\phishing\phishing\data\URLs.csv"

    igMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/ig.csv"
    csMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/cs.csv"
    rffMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/rff.csv"

    outputWin = r"C:\Users\lbustio\Documents\PycharmProjects\phishing\phishing\data\FeaturesDataset.csv"
    outputMac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/FeaturesRanking.csv"

    igFile = igMac
    csFile = csMac
    rffFile = rffMac

    output = outputMac


    print("#> Reading features ranking files...")

    ig = read_csv_file(igFile)
    rows, cols = ig.shape
    feat_names = ig[1:rows,[1,2]]

    ig = ig[1:rows,[0,1]].astype(float)
    ig_mean = statistics.mean(ig[:,0])
    ig_filtered = ig[ig[:,0] >= ig_mean,:]
    ig_features = ig_filtered[:,1]

    cs = read_csv_file(csFile)
    rows, cols = cs.shape
    cs = cs[1:rows, [0, 1]].astype(float)
    cs_mean = statistics.mean(cs[:, 0])
    cs_filtered = cs[cs[:, 0] >= cs_mean, :]
    cs_features = cs_filtered[:, 1]
    cs_min, cs_max = min(cs_filtered[:,0]), max(cs_filtered[:,0])
    for i, val in enumerate(cs_filtered[:,0]):
        cs_filtered[i,0] = (val - cs_min) / (cs_max - cs_min)

    rff = read_csv_file(rffFile)
    rows, cols = rff.shape
    rff = rff[1:rows, [0, 1]].astype(float)
    rff_mean = statistics.mean(rff[:, 0])
    rff_filtered = rff[rff[:, 0] >= rff_mean, :]
    rff_features = rff_filtered[:, 1]

    features = list(set(ig_features) & set(cs_features) & set(rff_features))

    result = []
    for i in features:
        ig_ranking = ig_filtered[np.where(ig_filtered[:,1] == i), 0]
        cs_ranking = cs_filtered[np.where(cs_filtered[:, 1] == i), 0]
        rff_ranking = rff_filtered[np.where(rff_filtered[:, 1] == i), 0]
        ranking_ave = float((ig_ranking + cs_ranking + rff_ranking) / 3)
        val = feat_names[feat_names[:, 0] == str(int(i)), 1]
        rank_row = [ranking_ave, i, val[0].astype(str)]
        result.append(rank_row)

    result = np.array(result)
    ind = np.argsort(result[:, 0])
    rind = ind[::-1]
    matrix = result[rind]

    write_csv_file(output, matrix)
    print("#> Done!!!")




if __name__ == "__main__":
    main()
