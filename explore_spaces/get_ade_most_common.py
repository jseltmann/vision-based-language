import pickle
from collections import defaultdict
import numpy as np
import codecs
import os
import json
from scipy.stats import spearmanr
from nltk.stem import WordNetLemmatizer


def get_spearman_corrs():
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/jseltmann/data/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()
    cat_num = defaultdict(int)
    all_objs = set()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    for jpath in json_paths:
        cat = jpath.split("/")[-3]
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        obj_names = [o['name'] for o in annot['object']]
        for obj_name in obj_names:
            if not cat in cat_counter:
                cat_counter[cat] = defaultdict(int)
            cat_counter[cat][obj_name] += 1
            all_objs.add(obj_name)
        cat_num[cat] += 1

    tfidfs = dict()
    for obj_name in all_objs:
        df = 0
        tfs = dict()
        for cat in cat_counter:
            all_terms = sum(cat_counter[cat].values())
            obj_count = cat_counter[cat][obj_name]
            tf = obj_count / all_terms
            tfs[cat] = tf
            if obj_count > 0:
                df += 1
        if df > 0:
            idf = 1 + np.log(len(cat_counter) / df)
            for cat in cat_counter:
                if not cat in tfidfs:
                    tfidfs[cat] = dict()
                tfidfs[cat][obj_name] = tfs[cat] * idf

    ordered_vs = dict()
    for cat in sorted(cat_counter.keys()):
        tfs = tfidfs[cat].items()
        obj_tfidfs = sorted(list(tfs), key=lambda p: p[1], reverse=True)
        objs_ordered = [p[0] for p in obj_tfidfs]
        ordered_vs[cat] = objs_ordered[:50]

    for c1, c2 in [(c1,c2) for c1 in ordered_vs for c2 in ordered_vs]:
        spear = spearmanr(ordered_vs[c1], ordered_vs[c2])
        print(c1, c2, spear)



def get_ade_cat_nums():
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/jseltmann/data/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)

    img_count = defaultdict(int)
    obj_count = defaultdict(int)
    dist_obj_per_img = defaultdict(int)
    dist_obj = defaultdict(set)

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    for jpath in json_paths:
        cat = jpath.split("/")[-3]
        img_count[cat] += 1
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        obj_names = [o['name'] for o in annot['object']]
        obj_count[cat] += len(obj_names)
        dist_obj_per_img[cat] += len(set(obj_names))
        for obj_name in obj_names:
            dist_obj[cat].add(obj_name)

    print("cat", "#imgs", "#objs", "objs/img", "#doi", "doi/img", "#doc")
    for cat in img_count:
        print(cat, img_count[cat], obj_count[cat], obj_count[cat]/img_count[cat], dist_obj_per_img[cat], dist_obj_per_img[cat]/img_count[cat], len(dist_obj[cat]))


def get_ade_frequencies_raw_names(print_tfidfs=False, obj_first=True):
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/jseltmann/data/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()
    cat_num = defaultdict(int)
    #obj_names = index["objectnames"]
    all_objs = set()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    for jpath in json_paths:
        cat = jpath.split("/")[-3]# + "/" + jpath.split("/")[-2]
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        #obj_names = [o['raw_name'] for o in annot['object']]
        for o in annot['object']:
            obj_names = o['name'].split(", ")
            obj_names.append(o['raw_name'])
            for obj_name in obj_names:
                if not cat in cat_counter:
                    cat_counter[cat] = defaultdict(int)
                cat_counter[cat][obj_name] += 1
                all_objs.add(obj_name)
        cat_num[cat] += 1

    if obj_first:
        count_reversed = dict()
        for cat in cat_counter:
            for obj in cat_counter[cat]:
                if not obj in count_reversed:
                    count_reversed[obj] = dict()
                if cat_counter[cat][obj] == 0:
                    print(cat,obj)
                count_reversed[obj][cat] = cat_counter[cat][obj]
        return count_reversed
    else:
        return cat_counter


def get_ade_scenes_frequencies_raw_names(obj_first=True):
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/jseltmann/data/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    for jpath in json_paths:
        cat = jpath.split("/")[-3]# + "/" + jpath.split("/")[-2]
        if not cat in cat_counter:
            cat_counter[cat] = dict()
        sce = jpath.split("/")[-2]
        if not sce in cat_counter[cat]:
            cat_counter[cat][sce] = defaultdict(int)
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        for o in annot['object']:
            obj_names = o['name'].split(", ")
            obj_names.append(o['raw_name'])
            for obj_name in obj_names:
                cat_counter[cat][sce][obj_name] += 1

    if obj_first:
        count_reversed = dict()
        for cat in cat_counter:
            count_reversed[cat] = dict()
            for sce in cat_counter[cat]:
                for obj in cat_counter[cat][sce]:
                    if not obj in count_reversed[cat]:
                        count_reversed[cat][obj] = dict()
                    count_reversed[cat][obj][sce] = cat_counter[cat][sce][obj]
        return count_reversed
    else:
        return cat_counter


def get_ade_scenes_tfidf_raw_names(obj_first=True, filter_animacy=False, filter_plurals=False):
    index_path = "/home/johann/Studium/MA/datasets/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/johann/Studium/MA/datasets/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()
    cat_num = dict()#defaultdict(int)
    #obj_names = index["objectnames"]
    all_objs = dict()#set()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    wnl = WordNetLemmatizer()

    for jpath in json_paths:
        cat = jpath.split("/")[-3]# + "/" + jpath.split("/")[-2]
        if not cat in all_objs:
            all_objs[cat] = set()
        sce = jpath.split("/")[-2]
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        #obj_names = [o['raw_name'] for o in annot['object']]
        for o in annot['object']:
            if filter_animacy and o['name_ndx'] in [1831, 29]:
                continue
            obj_names = o['name'].split(", ")
            obj_names.append(o['raw_name'])
            for obj_name in obj_names:
                if filter_plurals:
                    lemma = wnl.lemmatize(obj_name, "n")
                    if lemma != obj_name:
                        # sort out plurals
                        continue
                if not cat in cat_counter:
                    cat_counter[cat] = dict()
                if not sce in cat_counter[cat]:
                    cat_counter[cat][sce] = defaultdict(int)
                cat_counter[cat][sce][obj_name] += 1
                all_objs[cat].add(obj_name)
        if not cat in cat_num:
            cat_num[cat] = defaultdict(int)
        cat_num[cat][sce] += 1

    tfidfs = dict()
    for cat in all_objs:
        tfidfs[cat] = dict()
        for obj_name in all_objs[cat]:
            df = 0
            if obj_first:
                tfidfs[cat][obj_name] = dict()
            tfs = dict()
            for sce in cat_counter[cat]:
                all_terms = sum(cat_counter[cat][sce].values())
                obj_count = cat_counter[cat][sce][obj_name]
                tf = obj_count / all_terms
                tfs[sce] = tf
                if obj_count > 0:
                    df += 1
            if df > 0:
                idf = 1 + np.log(len(cat_counter[cat]) / df)
                for sce in cat_counter[cat]:
                    #if not cat in tfidfs:
                    #    tfidfs[cat] = dict()
                    #tfidfs[cat][obj_name] = tfs[cat] * idf
                    if obj_first:
                        tfidfs[cat][obj_name][sce] = tfs[sce] * idf
                    else:
                        if not sce in tfidfs[cat]:
                            tfidfs[cat][sce] = dict()
                        tfidfs[cat][sce][obj_name] = tfs[sce] * idf
    return tfidfs


def get_ade_tfidf_obj_part(filter_animacy=False, filter_plurals=False):
    index_path = "/home/johann/Studium/MA/datasets/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/johann/Studium/MA/datasets/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    obj_counter = dict()
    obj_num = defaultdict(int)
    all_parts = set()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    wnl = WordNetLemmatizer()

    for jpath in json_paths:
        #cat = jpath.split("/")[-3]# + "/" + jpath.split("/")[-2]
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        #obj_names = [o['raw_name'] for o in annot['object']]
        for o in annot['object']:
            name = o['name']
            parts = o['parts']['hasparts']
            if isinstance(parts, int):
                parts = [parts]
            for pnum in parts:
                if pnum >= len(annot['object']):
                    continue
                part = annot['object'][pnum]
                if filter_animacy and part['name_ndx'] in [1831, 29]:
                    continue
                pname = part['name']
                if filter_plurals:
                    lemma = wnl.lemmatize(pname, "n")
                    if lemma != pname:
                        # sort out plurals
                        continue
                if not name in obj_counter:
                    obj_counter[name] = defaultdict(int)
                obj_counter[name][pname] += 1
                all_parts.add(pname)
            obj_num[name] += 1

    tfidfs = dict()
    for part_name in all_parts:
        df = 0
        tfs = dict()
        for obj in obj_counter:
            all_terms = sum(obj_counter[obj].values())
            part_count = obj_counter[obj][part_name]
            tf = part_count / all_terms
            tfs[obj] = tf
            if part_count > 0:
                df += 1
        if df > 0:
            idf = 1 + np.log(len(obj_counter) / df)
            for obj in obj_counter:
                #if obj_first:
                #    tfidfs[obj_name][cat] = tfs[cat] * idf
                #else:
                if not obj in tfidfs:
                    tfidfs[obj] = dict()
                tfidfs[obj][part_name] = tfs[obj] * idf

    return tfidfs, obj_num


def get_ade_tfidf_raw_names(print_tfidfs=False, obj_first=True):
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/jseltmann/data/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()
    cat_num = defaultdict(int)
    #obj_names = index["objectnames"]
    all_objs = set()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    for jpath in json_paths:
        cat = jpath.split("/")[-3]# + "/" + jpath.split("/")[-2]
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        #obj_names = [o['raw_name'] for o in annot['object']]
        for o in annot['object']:
            obj_names = o['name'].split(", ")
            obj_names.append(o['raw_name'])
            for obj_name in obj_names:
                if not cat in cat_counter:
                    cat_counter[cat] = defaultdict(int)
                cat_counter[cat][obj_name] += 1
                all_objs.add(obj_name)
        cat_num[cat] += 1

    tfidfs = dict()
    for obj_name in all_objs:
        df = 0
        if obj_first:
            tfidfs[obj_name] = dict()
        tfs = dict()
        for cat in cat_counter:
            all_terms = sum(cat_counter[cat].values())
            obj_count = cat_counter[cat][obj_name]
            tf = obj_count / all_terms
            tfs[cat] = tf
            if obj_count > 0:
                df += 1
        if df > 0:
            idf = 1 + np.log(len(cat_counter) / df)
            for cat in cat_counter:
                #if not cat in tfidfs:
                #    tfidfs[cat] = dict()
                #tfidfs[cat][obj_name] = tfs[cat] * idf
                if obj_first:
                    tfidfs[obj_name][cat] = tfs[cat] * idf
                else:
                    if not cat in tfidfs:
                        tfidfs[cat] = dict()
                    tfidfs[cat][obj_name] = tfs[cat] * idf

    if print_tfidfs:
        i = 0
        for obj in tfidfs:
            i += 1
            if i == 10:
                break
            for cat in tfidfs[obj]:
                print(obj, cat, tfidfs[obj][cat])
    #if print_tfidfs:
    #    for cat in sorted(cat_counter.keys()):
    #        print(cat, cat_num[cat])
    #        obj_counts = cat_counter[cat].items()
    #        obj_counts = sorted(list(obj_counts), key=lambda p: p[1], reverse=True)
    #        tfs = tfidfs[cat].items()
    #        obj_tfidfs = sorted(list(tfs), key=lambda p: p[1], reverse=True)
    #        for obj_count, obj_tfidf in zip(obj_counts[:8], obj_tfidfs[:8]):
    #            #obj1, count = obj_count
    #            obj2, tfidf = obj_tfidf
    #            #print(obj, ":", count, tfidf)
    #            #print(obj1, "\t\t", obj2)
    #            print(obj2, tfidf)
    #        print("\n\n")
    return tfidfs


def get_ade_tfidf_full_names():
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    data_base_path = "/home/jseltmann/data/"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()
    cat_num = defaultdict(int)
    #obj_names = index["objectnames"]
    all_objs = set()

    paths = index["folder"]
    fns = index["filename"]
    file_paths = [os.path.join(p,fn) for p,fn in zip(paths,fns)]
    json_paths = [p.split(".")[0] + ".json" for p in file_paths]

    for jpath in json_paths:
        cat = jpath.split("/")[-3]# + "/" + jpath.split("/")[-2]
        jpath = os.path.join(data_base_path, jpath)
        with codecs.open(jpath, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        obj_names = [o['name'] for o in annot['object']]
        for obj_name in obj_names:
            if not cat in cat_counter:
                cat_counter[cat] = defaultdict(int)
            cat_counter[cat][obj_name] += 1
            all_objs.add(obj_name)
        cat_num[cat] += 1

    tfidfs = dict()
    for obj_name in all_objs:
        df = 0
        tfs = dict()
        for cat in cat_counter:
            all_terms = sum(cat_counter[cat].values())
            obj_count = cat_counter[cat][obj_name]
            tf = obj_count / all_terms
            tfs[cat] = tf
            if obj_count > 0:
                df += 1
        if df > 0:
            idf = 1 + np.log(len(cat_counter) / df)
            for cat in cat_counter:
                if not cat in tfidfs:
                    tfidfs[cat] = dict()
                tfidfs[cat][obj_name] = tfs[cat] * idf

    for cat in sorted(cat_counter.keys()):
        print(cat, cat_num[cat])
        obj_counts = cat_counter[cat].items()
        obj_counts = sorted(list(obj_counts), key=lambda p: p[1], reverse=True)
        tfs = tfidfs[cat].items()
        obj_tfidfs = sorted(list(tfs), key=lambda p: p[1], reverse=True)
        for obj_count, obj_tfidf in zip(obj_counts[:8], obj_tfidfs[:8]):
            #obj1, count = obj_count
            obj2, tfidf = obj_tfidf
            #print(obj, ":", count, tfidf)
            #print(obj1, "\t\t", obj2)
            print(obj2, tfidf)
        print("\n\n")

    #print("-------------")
    #for cat, catdict in cat_counter.items():
    #    #if not "person, individual, someone, somebody, mortal, soul" in catdict:
    #    #    print(cat)
    #    keys = catdict.keys()
    #    sw = [k.startswith("person, individual") for k in keys]
    #    if not True in sw:
    #        print(cat)
        #keysl = sorted(keys, key=len, reverse=True)
        #for k in keysl:
        #    if k.startswith("person, individual"):
        #        print(cat)


def get_tfidf_obj_names():
    #ade_path = "/home/jseltmann/data/ADE20k_2021_17_01/images/ADE/training"
    index_path = "/home/jseltmann/data/ADE20K_2021_17_01/index_ade20k.pkl"
    with open(index_path, "rb") as indf:
        index = pickle.load(indf)
    cat_counter = dict()
    obj_names = index["objectnames"]

    #for cat, obj_counts in zip(index["folder"], index["objectPresence"]):
    cats = index["folder"]
    cats = [path.split("/")[-2] for path in cats]
    for i, obj_counts in enumerate(index["objectPresence"]):
        obj_name = obj_names[i]
        #for cat, obj_count in zip(index["folder"], obj_counts):
        for cat, obj_count in zip(cats, obj_counts):
            if not cat in cat_counter:
                cat_counter[cat] = defaultdict(int)
        #for i, objc in enumerate(obj_counts):
        #    if objc > 0:
        #        obj_name = obj_names[i]
            cat_counter[cat][obj_name] += obj_count

    tfidfs = dict()
    for obj_name in obj_names:
        df = 0
        tfs = dict()
        for cat in cat_counter:
            #if not cat in tfs:
            #    tfs[cat] = dict()
            all_terms = sum(cat_counter[cat].values())
            obj_count = cat_counter[cat][obj_name]
            tf = obj_count / all_terms
            tfs[cat] = tf
            if obj_count > 0:
                df += 1
        if df > 0:
            idf = np.log(len(cat) / df)
            for cat in cat_counter:
                if not cat in tfidfs:
                    tfidfs[cat] = dict()
                tfidfs[cat][obj_name] = tfs[cat] * idf

    for cat in cat_counter:
        print(cat)
        obj_counts = cat_counter[cat].items()
        obj_counts = sorted(list(obj_counts), key=lambda p: p[1], reverse=True)
        tfs = tfidfs[cat].items()
        obj_tfidfs = sorted(list(tfs), key=lambda p: p[1], reverse=True)
        for obj_count, obj_tfidf in zip(obj_counts[:20], obj_tfidfs[:20]):
            obj1, count = obj_count
            obj2, tfidf = obj_tfidf
            print(obj2, ":", tfidf)
            #print(obj1, "\t\t", obj2)
        print("\n\n")


if __name__=="__main__":
    #get_spearman_corrs()
    #get_tfidf_obj_names()
    tfidfs = get_ade_tfidf_raw_names(print_tfidfs=True)
    with open("tfidfs.json", "w") as tf:
        json.dump(tfidfs, tf)
