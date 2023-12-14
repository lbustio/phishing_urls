import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# Plotting routine
def plot(dataset):
    sns.set(color_codes=True)
    sns.set(rc={"figure.figsize": (8, 4)})
    plt.plot(range(len(dataset)), list(dataset.values()))
    plt.xticks(range(len(dataset)), list(dataset.keys()), rotation=45)
    plt.title("Training results")
    plt.legend(loc="upper right")
    plt.xlabel("Algorithms")
    plt.ylabel("Classification score")
    plt.show()


# Classification step
def classify(urls, labels, k_fold):
    print("#> Starting the training process...")
    rows, cols = urls.shape
    urls = urls[1: rows,:]
    labels = labels[1: rows]

    model = {"KNN": KNeighborsClassifier(),
             "RandomForest": RandomForestClassifier(n_estimators=100),
             "DecisionTree": DecisionTreeClassifier(max_depth=10),
             "SVM": SVC(gamma="scale")
             }

    results = {}
    if k_fold == 1:
        # Not crossvalidate the data:
        x_train, x_test, y_train, y_test = train_test_split(urls, labels, test_size = 0.3)

    for algo in model:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        start_time = time.time()
        print("#>   Testing %s. Started at %s...." % (algo, current_time))

        clf = model[algo]
        if k_fold > 1:
            print("#>     Crossvalidating the data with %s folds...." % (k_fold))
            # if crossvalidating the data:
            score = cross_val_score(clf, urls, labels, cv = k_fold)
            print("#>     Accuracy: %f (+/- %f)" % (score.mean(), score.std() * 2))
            results[algo] = score.mean()
        else:
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            print("#>    %s : %s " % (algo, score))
            results[algo] = score
            results[algo] = score
            res = clf.predict(x_test)
            mt = confusion_matrix(y_test, res)
            print("#>   Confussion matrix:")
            print(mt)
            print("#>   Prediction summary")
            tp = mt[0][0]
            fp = mt[0][1]
            tn = mt[1][1]
            fn = mt[1][0]
            print("#>   False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
            print("#>   False negative rate : %f %%" % ((mt[1][0] / float(sum(mt[1])) * 100)))
            print("#>   Precision : %f %%" % ((tp / float(tp + fp))))
            print("#>   Recall : %f %%" % ((tp / float(tp + fn))))
            print("#>   F-measure : %f %%" % (2 * ((tp / float(tp + fp)) * (tp / float(tp + fn))) / float(
                (tp / float(tp + fp)) + (tp / float(tp + fn)))))
            print("#>   Accuracy : %f %%" % ((tp + tn) / (tp + tn + fn + fp)))

        print("#>     Time needed: %s seconds" % (time.time() - start_time))

    print("#> Ending the training process...")

    plot(results)

    winner = max(results, key = results.get)
    clf = model[winner]
    print("#> The best results were obtained using %s with %f of score" % (winner, results[winner]))
    if k_fold == 1:
        print("#> Classifying unknown data using %s" % winner)
        res = clf.predict(x_test)
        mt = confusion_matrix(y_test, res)
        print("#>   Confussion matrix:")
        print(mt)
        print("#>   Prediction summary")
        tp = mt[0][0]
        fp = mt[0][1]
        tn = mt[1][1]
        fn = mt[1][0]
        print("#>   False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
        print("#>   False negative rate : %f %%" % ((mt[1][0] / float(sum(mt[1])) * 100)))
        print("#>   Precision : %f %%" % ((tp / float(tp + fp))))
        print("#>   Recall : %f %%" % ((tp / float(tp + fn))))
        print("#>   F-measure : %f %%" % (2*((tp / float(tp + fp)) * (tp / float(tp + fn)))/float((tp / float(tp + fp)) + (tp / float(tp + fn)))))
        print("#>   Accuracy : %f %%" % ((tp + tn)/(tp + tn + fn + fp)))

    return clf
    # end!

def classify_individual(urls, labels, k_fold):
    print("#> Starting the training process for individual features...")
    rows, cols = urls.shape
    urls = urls[1: rows, :]
    labels = labels[1: rows]

    model = {"KNN": KNeighborsClassifier(),
             "RandomForest": RandomForestClassifier(n_estimators=100),
             "DecisionTree": DecisionTreeClassifier(max_depth=10),
             "SVM": SVC(gamma="scale")
             }

    results = {}

    for feature in range(0, cols):
        if k_fold == 1:
            # Not crossvalidate the data:
            x_train, x_test, y_train, y_test = train_test_split(urls[:, feature], labels, test_size=0.3)
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

        for algo in model:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            start_time = time.time()
            print("#>   Testing feature %s with %s. Started at %s...." % (feature, algo, current_time))

            clf = model[algo]
            if k_fold > 1:
                print("#>     Crossvalidating the data with %s folds...." % (k_fold))
                # if crossvalidating the data:
                feat = urls[:, feature]
                feat = feat.reshape(-1, 1)
                score = cross_val_score(clf, feat, labels, cv=k_fold)
                print("#>     Accuracy: %f (+/- %f)" % (score.mean(), score.std() * 2))
                results[algo] = score.mean()
            else:
                clf.fit(x_train, y_train)
                score = clf.score(x_test, y_test)
                print("#>    %s : %s " % (algo, score))
                results[algo] = score

                print("#>     Time needed: %s seconds" % (time.time() - start_time))

        print("#> Ending the training process for feature %s" % (feature))

        plot(results)

        winner = max(results, key=results.get)
        clf = model[winner]
        print("#> The best results were obtained using %s with %f of score" % (winner, results[winner]))
        if k_fold == 1:
            print("#> Classifying unknown data using %s" % winner)
            res = clf.predict(x_test)
            mt = confusion_matrix(y_test, res)
            print("#>   Confussion matrix:")
            print(mt)
            print("#>   Prediction summary")
            tp = mt[0][0]
            fp = mt[0][1]
            tn = mt[1][1]
            fn = mt[1][0]
            print("#>   False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
            print("#>   False negative rate : %f %%" % ((mt[1][0] / float(sum(mt[1])) * 100)))
            print("#>   Precision : %f %%" % ((tp / float(tp + fp))))
            print("#>   Recall : %f %%" % ((tp / float(tp + fn))))
            print("#>   F-measure : %f %%" % (2 * ((tp / float(tp + fp)) * (tp / float(tp + fn))) / float(
                    (tp / float(tp + fp)) + (tp / float(tp + fn)))))
            print("#>   Accuracy : %f %%" % ((tp + tn) / (tp + tn + fn + fp)))

    print("#> Ending the training process!!!")


def classify_cummulative(urls, labels, k_fold):
    print("#> Starting the training process for individual features...")
    rows, cols = urls.shape
    urls = urls[1: rows, :]
    labels = labels[1: rows]

    model = {"KNN": KNeighborsClassifier(),
             "RandomForest": RandomForestClassifier(n_estimators=100),
             "DecisionTree": DecisionTreeClassifier(max_depth=10),
             "SVM": SVC(gamma="scale")
             }

    results = {}

    for feature in range(1, (cols + 1)):
        if k_fold == 1:
            # Not crossvalidate the data:
            x_train, x_test, y_train, y_test = train_test_split(urls[:,0:feature], labels, test_size = 0.3)
            if feature == 1:
                x_train = x_train.reshape(-1, 1)
                x_test = x_test.reshape(-1, 1)


        for algo in model:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            start_time = time.time()
            print("#>   Testing features 0-%s with %s. Started at %s...." % (feature, algo, current_time))

            clf = model[algo]
            if k_fold > 1:
                print("#>     Crossvalidating the data with %s folds...." % (k_fold))
                # if crossvalidating the data:
                feat = urls[:,0:feature]
                if feature == 1:
                    feat = feat.reshape(-1, 1)
                score = cross_val_score(clf, feat, labels, cv = k_fold)
                print("#>     Accuracy: %f (+/- %f)" % (score.mean(), score.std() * 2))
                results[algo] = score.mean()
            else:
                clf.fit(x_train, y_train)
                score = clf.score(x_test, y_test)
                print("#>    %s : %s " % (algo, score))

            print("#>     Time needed: %s seconds" % (time.time() - start_time))

        print("#> Ending the training process for feature %s", (feature))

        plot(results)

        winner = max(results, key = results.get)
        clf = model[winner]
        print("#> The best results were obtained using %s with %f of score" % (winner, results[winner]))
        if k_fold == 1:
            print("#> Classifying unknown data using %s" % winner)
            res = clf.predict(x_test)
            mt = confusion_matrix(y_test, res)
            print("#>   Confussion matrix:")
            print(mt)
            print("#>   Prediction summary")
            tp = mt[0][0]
            fp = mt[0][1]
            tn = mt[1][1]
            fn = mt[1][0]
            print("#>   False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
            print("#>   False negative rate : %f %%" % ((mt[1][0] / float(sum(mt[1])) * 100)))
            print("#>   Precision : %f %%" % ((tp / float(tp + fp))))
            print("#>   Recall : %f %%" % ((tp / float(tp + fn))))
            print("#>   F-measure : %f %%" % (2*((tp / float(tp + fp)) * (tp / float(tp + fn)))/float((tp / float(tp + fp)) + (tp / float(tp + fn)))))
            print("#>   Accuracy : %f %%" % ((tp + tn)/(tp + tn + fn + fp)))

    print("#> Ending the training process!!!")