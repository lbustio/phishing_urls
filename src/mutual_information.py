import csv
import os
import re
import math


def filefromCorpus(corpusOriginal, tipo=["txt"]):
    salida = list()
    ruta = corpusOriginal
    for root, dirs, files in os.walk(ruta):

        for file in [f for f in files]:
            print(os.path.join(root, file))

            ruta = str(os.path.join(root, file))
            if ruta.split(".")[-1] in tipo:
                salida.append(os.path.join(root, file))
    return salida


def classesFromCorpus(archivos):
    users = list()
    path = archivos
    for base, dirs, files in os.walk(path):
        users = dirs
        break
    return users


def convert2Dict(a):
    res = dict()
    for element in a:
        res[element] = dict()
    return res


def informacionMutua(corpus):

    archivoSalidaRasgo = corpus + "/mutualInformation.csv"
    rasgos = classesFromCorpus(corpus)
    regexp = u"[\\w]+"
    patter = re.compile(regexp)

    for rasgo in rasgos:
        corpusImagenes = corpus + "/" + rasgo
        archivoSalida = archivoSalidaRasgo + "_" + rasgo + ".csv"
        #clases = convert2Dict(classesFromCorpus(corpusImagenes))
        clases = convert2Dict(classesFromCorpus(corpus))

        print(str(rasgo))

        archivos = filefromCorpus(corpusImagenes)

        probabilidad_Clase = dict()
        for archivo in archivos:
            if not archivo.split("/")[-2] in probabilidad_Clase:
                probabilidad_Clase[archivo.split("/")[-2]] = 0
            probabilidad_Clase[archivo.split("/")[-2]] += 1
        palabrasCantidad = dict()
        for archivo in archivos:
            leer = open(archivo, "r")
            clase = archivo.split("/")[-2]
            conjunto = set()
            for line in leer:
                palabras = patter.findall(line)
                palabrasMinusculas = list()
                for palabra in palabras:
                    palabrasMinusculas.append(palabra.lower())
                palabras = palabrasMinusculas
                conjunto |= set(palabras)
            leer.close()
            for palabraInConjunto in conjunto:
                if not palabraInConjunto in palabrasCantidad:
                    palabrasCantidad[palabraInConjunto] = 0
                palabrasCantidad[palabraInConjunto] += 1

            for palabra in conjunto:
                if not palabra in clases[clase]:
                    clases[clase][palabra] = 0
                clases[clase][palabra] += 1


        writer = csv.writer(open(archivoSalida, "w"), delimiter=',')
        for clase in clases.keys():
            writer.writerow(["Class", clase])
            writer.writerow(["Palabra", "Mutual Information", "Repeticiones", "Valor"]);
            for palabra in clases[clase].keys():
                total_palabra = 0
                for claseCount in clases.keys():
                    if palabra in clases[claseCount]:
                        total_palabra += clases[claseCount][palabra]

                p_conjunta = clases[clase][palabra] / float(len(archivos))
                p_palabra = total_palabra / float(len(archivos))
                p_clase = probabilidad_Clase[clase] / float(len(archivos))

                mutual_informationRes = math.log(p_conjunta / (p_palabra * p_clase), 2)
                if palabrasCantidad[palabra] > 10:
                    writer.writerow([u"".join(palabra), mutual_informationRes, palabrasCantidad[palabra],
                                     mutual_informationRes * palabrasCantidad[palabra]])