import math
import os.path
import re
import string
import numpy as np
import tldextract
import yake

from urllib.parse import parse_qs
from urllib.parse import urlparse
from src.files import read_mutual_information_file
from src.mutual_information import informacionMutua


# tokenize the input url.
def tokenizer(url):
    pattern = "\\W"
    result = re.split(pattern, url)
    result = [x for x in result if x]
    # result = " ".join(result)
    return result


# calculates the length of the input url.
def url_length(url):
    return len(url)


# counts the number of digits in the url.
def digit_counter(url):
    numbers = sum(c.isdigit() for c in url)
    return numbers


# count the number of letters in the url.
def letter_counter(url):
    letters = sum(c.isalpha() for c in url)
    return letters


# count the number of subdomains in the url.
def subdomains_tokens_counter(url):
    tokens = tldextract.extract(url)
    return len(tokenizer(tokens.subdomain))


# Gets the length of the subdomain in the url.
def subdomains_length(url):
    tokens = tldextract.extract(url)
    result = len(tokens.subdomain) - tokens.subdomain.count(".")
    return result


# count the number of tokens in the domain of the url.
def domain_tokens_counter(url):
    tokens = tldextract.extract(url)
    return len(tokenizer(tokens.domain))


# count the number of tokens in the domain of the url.
def domain_length(url):
    tokens = tldextract.extract(url)
    result = len(tokens.domain) - tokens.domain.count(".")
    return result


# count the number of tokens in the TLD of the url.
def tld_tokens_counter(url):
    tokens = tldextract.extract(url)
    return len(tokenizer(tokens.suffix))


# count the length of the TLD
def tld_length(url):
    tokens = tldextract.extract(url)
    result = len(tokenizer(tokens.suffix))
    return result


# count the number of characters in the hostname of the url.
def hostname_length(url):
    ext = tldextract.extract(url)
    result = len(ext.subdomain) - ext.subdomain.count(".") + len(ext.domain) - ext.domain.count(".") + \
             len(ext.suffix) - ext.suffix.count(".")
    return result


# count the number of slashes in the url.
def slash_counter(url):
    result = url.count("/") - 2
    return result


# counts the number of special chars in the url.
def hostname_spchar_counter(url):
    ext = tldextract.extract(url)
    hostname = ext.fqdn
    hostname = hostname.replace(".", "")
    hostname = hostname.replace("/", "")
    special = 0
    for i in range(len(hostname)):
        if ((not hostname[i].isalpha()) and (not hostname[i].isdigit())):
            special = special + 1
    return special


# Find if there is an IP address in the url
def match_ip_addess(url):
    ip = re.compile('(([2][5][0-5]\.)|([2][0-4][0-9]\.)|([0-1]?[0-9]?[0-9]\.)){3}'
                    + '(([2][5][0-5])|([2][0-4][0-9])|([0-1]?[0-9]?[0-9]))')

    match = ip.search(url)
    if (match):
        return 1
    else:
        return 0


# Count the number of unicode chars in the url
def count_unicodes(url):
    cant_unicode = 0
    for each in url:
        if (ord(each) > 128):
            cant_unicode += 1
    return cant_unicode


# Determine if the url is HTTPS or HTTP
def is_https(url):
    t = urlparse(url)
    if (t.scheme == "https"):
        return 1
    else:
        return 0


# count the number of dots in the url
def url_dots_counter(url):
    return url.count(".")


# Get the number of hyphens in the hostname of the url
def url_hyphen_counter(url):
    ext = tldextract.extract(url)
    hostname = ext.subdomain + ext.domain + ext.suffix
    return hostname.count("-")


# count the number of parameters in query
def query_params_counter(url):
    query_params = parse_qs(url)
    return len(query_params.keys())


# Entropy (H) calculation
def H(data):
    printables = (ord(c) for c in string.printable)
    if not data:
        return 0
    entropy = 0
    for x in printables:
        p_x = float(data.count(chr(x))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy


# Calculate the entropy of the hostname
def hostname_entropy_calculator(url):
    ext = tldextract.extract(url)
    return H(ext.fqdn)


# Calculate the entropy of the complete url
def url_entropy_calculator(url):
    return H(url)


# Calculate the entropy of the path of the url
def path_entropy_calculator(url):
    ssl = ""
    if (is_https(url)):
        ssl = "https://"
    else:
        ssl = "http://"

    ext = tldextract.extract(url)
    path = url.replace(ext.fqdn, "")
    path = path.replace(ssl, "")
    return H(path)


# calculates the rate between the domain and the url
def domain_url_rate_calculator(url):
    ext = tldextract.extract(url)
    return len(ext.domain) / len(url)


# calculates the rate between the subdomain and the url
def subdomain_url_rate_calculator(url):
    ext = tldextract.extract(url)
    return len(ext.subdomain) / len(url)


# calculates the rate between the hostname and the url
def path_url_rate_calculator(url):
    return len(urlparse(url).path) / len(url)


# calculates the rate between the path and the url
def hostname_url_rate_calculator(url):
    ext = tldextract.extract(url)
    return len(ext.fqdn) / len(url)


# calculates the rate between the args and the url
def args_url_rate_calculator(url):
    query = urlparse(url).query
    args = parse_qs(query)
    args = "".join(args.keys())
    return len(args) / len(url)


# calculates the rate between the path and the domain
def path_domain_rate_calculator(url):
    p = urlparse(url).path
    h = urlparse(url).hostname
    if p is not None:
        lp = len(p)
    else:
        lp = 0
    if h is not None:
        lh = len(h)
    else:
        lp = 0
        lh = 1

    return lp / lh


# calculates the rate between the args and domain
def args_domain_rate_calculator(url):
    la = 0
    ld = 1
    query = urlparse(url).query
    if query is not None:
        args = parse_qs(query)
        if args is not None:
            args = "".join(args.keys())
            la = len(args)

    ext = tldextract.extract(url)
    if ext is not None:
        d = ext.domain
        if d is not None:
            ld = len(d)
            if ld == 0:
                la = 0
                ld = 1

    return la / ld


# calculates the rate between the args and domain
def args_path_rate_calculator(url):
    query = urlparse(url).query
    args = parse_qs(query)
    args = "".join(args.keys())
    l = len(urlparse(url).path)
    if (l > 0):
        rate = len(args) / l
    else:
        rate = 0

    return rate


# count the number of letter-digit-letter sequences
def letter_digit_letter_counter(url):
    return len(re.findall("\w\d\w", url))


# count the number of digit-letter-digit sequences
def digit_letter_digit_counter(url):
    result = len(re.findall("\d\w\d", url))
    return result


# get the length of the longest item in the url
def longest_item_length(url):
    elements = tokenizer(url)
    return len(max(elements, key=len))


# get the length of the shortes item in the url
def shortest_item_length(url):
    elements = tokenizer(url)
    return len(min(elements, key=len))


# Count the number of subdirectories in the url
def subdirs_counter(url):
    path = urlparse(url).path
    subdirs = os.path.split(path)
    sd = []
    if (subdirs[0] != ""):
        sd = subdirs[0].split("/")
        sd = [i for i in sd if i]

    return len(sd)


# calculate the average of the token's length in a url.
def tokens_ave_length_calculator(url):
    strings = tokenizer(url)
    total_avg = sum(map(len, strings)) / len(strings)
    return total_avg


# calculate the rate between number of letters and the length of the hostname
def letter_hostname_rate_calculator(url):
    ext = tldextract.extract(url)
    return letter_counter(ext.fqdn) / len(url)


# calculate the rate between number of letters and the length of the hostname
def digit_hostname_rate_calculator(url):
    ext = tldextract.extract(url)
    return digit_counter(ext.fqdn) / len(url)


# calculate the rate between number of symbols and the length of the hostname
def symbol_hostname_rate_calculator(url):
    return hostname_spchar_counter(url) / len(url)


# count the number of digits in the hostname
def digit_hostname_counter(url):
    ext = tldextract.extract(url)
    return digit_counter(ext.fqdn)


# count the number of letters in the hostname
def letter_hostname_counter(url):
    ext = tldextract.extract(url)
    return letter_counter(ext.fqdn)


# count the number of symbols in the hostname
def symbol_hostname_counter(url):
    ext = tldextract.extract(url)
    return hostname_spchar_counter(ext.fqdn)


# determine if the url redirects to an executable file
def contains_executable(url):
    if (".exe" in url):
        return 1
    else:
        return 0


# get the port of the url
def get_port(url):
    port = urlparse(url).port
    if (port == None):
        port = -1
    return port


# Get the edition distance between two input strings
def edition_distance(str1, str2):
    d = dict()
    for i in range(len(str1) + 1):
        d[i] = dict()
        d[i][0] = i
    for i in range(len(str2) + 1):
        d[0][i] = i
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + (not str1[i - 1] == str2[j - 1]))
    return float(d[len(str1)][len(str2)]) / float(max(len(str1), len(str2)))


# get the mutual information for an url considering each class
def get_mutual_info_average(url, phishing, normal):
    mutual_info_accummulated_phishing = 0
    mutual_info_accummulated_normal = 0
    tokens = tokenizer(url)
    for token in tokens:
        phis_row = phishing[phishing[:, 0] == token]
        if len(phis_row) > 0:
            mutual_info_accummulated_phishing += float(phis_row[0, 1])

        normal_row = normal[normal[:, 0] == token]
        if len(normal_row) > 0:
            mutual_info_accummulated_normal += float(normal_row[0, 1])

    mi_phishing = mutual_info_accummulated_phishing / len(tokens)
    mi_normal = mutual_info_accummulated_normal / len(tokens)

    return mi_phishing, mi_normal


# calculate the edition distance
def similarity_rate(url, terms):
    result = 0
    tokens = tokenizer(url)
    if (len(tokens) > 0):
        sum = 0
        for token in tokens:
            min = -1
            for term in terms:
                sim = edition_distance(token, term)
                if (min < sim):
                    min = sim
            sum = sum + min
            result = sum / len(tokens)

    return result


# obtains keywords from a class instance
def keywords_extractor(urls, k, label):
    urls = urls[urls[:, 1] == label]
    urls = urls[:, 0]
    urls = " ".join(urls)
    custom_kwextractor = yake.KeywordExtractor(n=1, top=k)
    keywords = custom_kwextractor.extract_keywords(urls)
    keywords.sort(key=lambda tup: tup[0], reverse=True)
    return keywords


def keyword_count(keywords, url):
    tokens = tokenizer(url)
    if "http" in tokens:
        tokens.remove("http")

    if "https" in tokens:
        tokens.remove("https")

    kw_mat = np.array(keywords, dtype=None, copy=True, order='K', subok=False, ndmin=0)
    result = 0.0
    for t in tokens:
        tup = kw_mat[kw_mat[:, 1] == t]
        if tup.shape[0] > 0:
            result = result + float(tup[0][0])

    return result


# determine if the url is redirected
def detect_at(url):
    result = url.count("@")
    return result


# Calculates the mutial information for a given dataset
def calculate_mutual_information(urls, labels):
    index = 1
    base_path_mac = r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/Separated/Sep"
    base_path_win = ""
    base_path = base_path_mac
    for url in urls:
        print("#>   Processing URL %s of %s" % (index, len(labels)))

        ## Separando las URL para obtener la mutual distance

        line = tokenizer(url)
        text = " "
        text = text.join(line)
        writer = open(base_path + "/" + str(labels[index - 1]) + "/url_" + str(index) + ".txt", "w")
        writer.write(text + "\n")
        writer.close()
        index = index + 1

    # obtaining mutual information
    informacionMutua(base_path)
    print("#>   Mutual Information calculation done!")
    exit()


# Obtains the features vector for the urls.
def get_features(urls):
    # If MI is not calculated....
    # addresses = urls[:,0]
    # labels = urls[:,1]
    # calculate_mutual_information(addresses, labels)

    k = 1000
    # Obtaining the k=1000 keywords of each class
    print("#>   Extracting the top %s keywords for legitimates URL..." % k)
    legitimate_keywords = keywords_extractor(urls, k, "0")

    print("#>   Extracting the top %s keywords for phishing URL..." % k)
    phishing_keywords = keywords_extractor(urls, k, "1")

    labels = urls[:, 1]
    urls = urls[:, 0]
    dataset = [["url_len", "subdomain_len", "domain_len", "tld_len", "hostname_len", "longest_len", "shortest_len",
               "tokens_ave", "digit_count", "subdomains_tokens_count", "domain_tokens_count", "tld_tokens_count",
               "hostname_sp_chars_count", "slash_count", "unicode_count", "count_dots_in_url", "url_hyphen_count",
               "params_query_count", "subdirs_count", "digit_hostname_count", "letter_hostname_count",
               "symbol_hostname_count", "ip_addr", "is_SSL", "has_exe", "port", "cant_phishing_kw",
               "cant_legitimate_kw", "url_entropy", "hotname_entropy", "path_entropy", "mi_phishing", "mi_normal",
               "domain rate", "subdomain_rate", "hostname_rate", "path_rate", "arguments_rate", "path_domain_rate",
               "args_domain_rate", "args_path_rate", "ldl_count", "dld_count", "letter_hostname_rate",
               "digit_hostname_rate", "symbol_hostname_rate", "label"]]

    index = 0
    phishing, normal = read_mutual_information_file(
        r"/Users/lazarobustiomartinez/Documents/Proyectos/PyCharmProjects/phishing/data/Mutual_Info_Ordered.csv")

    for url in urls:
        print("#>   Processing URL %s of %s" % (index, len(labels)))
        #########################################
        # Length-related features:
        # F1: length of the url.
        url_len = url_length(url)

        # F2: Length of the subdomain of the url (number of characters).
        subdomain_len = subdomains_length(url)

        # F3: Length of the domain of the url (number of characters).
        domain_len = domain_length(url)

        # F4: Length of the top level domains (number of characters).
        tld_len = tld_length(url)

        # F5: Length of the hostname of the url (number of characters)
        hostname_len = hostname_length(url)

        # F6: longest item length in url
        longest_len = longest_item_length(url)

        # F7: shortest item length in url
        shortest_len = shortest_item_length(url)

        # HF8: token's length average
        tokens_ave = tokens_ave_length_calculator(url)

        #########################################
        # Counting-related features
        # F9: Number of digits in the url.
        digit_count = digit_counter(url)

        # F10: Numbers of tokens in the subdomain of the url.
        subdomains_tokens_count = subdomains_tokens_counter(url)

        # F11: Number of tokens in the domain of the url.
        domain_tokens_count = domain_tokens_counter(url)

        # F12: Number of top level domain in the url.
        tld_tokens_count = tld_tokens_counter(url)

        # F13: Number of special chars in the hostname.
        hostname_sp_chars_count = hostname_spchar_counter(url)

        # F14: Number of slashes in the url.
        slash_count = slash_counter(url)

        # F15: Number of unicode chars in the url.
        unicode_count = count_unicodes(url)

        # F16: count the number of dots in the url.
        count_dots_in_url = url_dots_counter(url)

        # F17: count the numbers of hyphen in the hostname of the url.
        url_hyphen_count = url_hyphen_counter(url)

        # F18 count the number of parameters in query
        params_query_count = query_params_counter(url)

        # F19: count the number of subdirs in the url
        subdirs_count = subdirs_counter(url)

        # F20: Number of digits in the hos.
        digit_hostname_count = digit_hostname_counter(url)

        # F21: Number of letters in the hos.
        letter_hostname_count = letter_hostname_counter(url)

        # F22: Number of symbols in the hos.
        symbol_hostname_count = symbol_hostname_counter(url)

        #########################################
        # HTTP/S-based features
        # F23: Detect if the url contains ip address.
        ip_addr = match_ip_addess(url)

        # F24: Determine if the url is HTTPS or HTTP.
        is_SSL = is_https(url)

        # F25: contains any executable file?
        has_exe = contains_executable(url)

        # F26: get the default port
        port = get_port(url)

        #########################################
        # Natural Language Processing-based features
        # F27: determining the importance of keywords in phishing
        cant_phishing_kw = keyword_count(phishing_keywords, url)

        # F28: count the number of class keywords
        # Determinar la cantidad de palabras clave que hay en la url segun la clase a la que pertence
        cant_legitimate_kw = keyword_count(legitimate_keywords, url)

        # F29: calculate the entropy of the url
        url_entropy = url_entropy_calculator(url)

        # F30: calculate the hostname entropy
        hotname_entropy = hostname_entropy_calculator(url)

        # F31: calculate the entropy of the path
        path_entropy = path_entropy_calculator(url)

        # F32 and F33: get mutual information average for phishing and et mutual information
        # average for legitimate
        mi_phishing, mi_normal = get_mutual_info_average(url, phishing, normal)

        #########################################
        # Rate-based features

        # F34: calculate the domain - url rate
        domain_rate = domain_url_rate_calculator(url)

        # F35: calculate the subdomain - url rate
        subdomain_rate = subdomain_url_rate_calculator(url)

        # F36: calculate the hostname - url rate
        hostname_rate = hostname_url_rate_calculator(url)

        # F37: calculate the path - url rate
        path_rate = path_url_rate_calculator(url)

        # F38: calculate the arguments - url rate
        arguments_rate = args_url_rate_calculator(url)

        # F39: calculate path - domain rate
        path_domain_rate = path_domain_rate_calculator(url)

        # F40: calculate args - domain rate
        args_domain_rate = args_domain_rate_calculator(url)

        # F41: calculate args - path rate
        args_path_rate = args_path_rate_calculator(url)

        # F42: Letter-digit-letter sequences counting
        ldl_count = letter_digit_letter_counter(url)

        # F43: Digit-letter-digit sequences counting
        dld_count = digit_letter_digit_counter(url)

        # F44: Letter-hostname rate
        letter_hostname_rate = letter_hostname_rate_calculator(url)

        # F45: Digit-hostname rate
        digit_hostname_rate = digit_hostname_rate_calculator(url)

        # F46: Symbols-hostname rate
        symbol_hostname_rate = symbol_hostname_rate_calculator(url)

        line = [url_len, subdomain_len, domain_len, tld_len, hostname_len, longest_len, shortest_len,
                tokens_ave, digit_count, subdomains_tokens_count, domain_tokens_count, tld_tokens_count,
                hostname_sp_chars_count, slash_count, unicode_count, count_dots_in_url, url_hyphen_count,
                params_query_count, subdirs_count, digit_hostname_count, letter_hostname_count,
                symbol_hostname_count, ip_addr, is_SSL, has_exe, port, cant_phishing_kw, cant_legitimate_kw,
                url_entropy, hotname_entropy, path_entropy, mi_phishing, mi_normal, domain_rate,
                subdomain_rate, hostname_rate, path_rate, arguments_rate, path_domain_rate, args_domain_rate,
                args_path_rate, ldl_count, dld_count, letter_hostname_rate, digit_hostname_rate,
                symbol_hostname_rate, labels[index]]

        dataset.append(line)
        index = index + 1

    mat = np.array(dataset)
    return mat
