
def translate_table(line, suite_pos):
    fields = line.split("|")
    suite_name = fields[suite_pos]
    fields = suite_name.split("_")
    other = []
    if fields[0] == "ade":
        template = "thereis"
        if fields[1] == "same":
            strategy = "ADE-SI"
            if fields[-1] == "tisce":
                other.append("sce")
            if fields[-1] == "ticat":
                other.append("cat")
        elif fields[1] == "diff":
            if fields[2] == "cat":
                strategy = "ADE-DC"
            else:
                strategy = "ADE-SC"
            if fields[-1] == "tisce":
                other.append("sce")
            if fields[-1] == "ticat":
                other.append("cat")
        if fields[1] == "tfidf" or fields[1] == "freq":
            s = fields[1][0].capitalize() + fields[1][1:]
            if fields[3] == "category":
                strategy = s + "-Cat"
            elif fields[3] == "scene":
                strategy = s + "-Sce"
            elif fields[3] == "obj":
                strategy = s + "-Obj"
            if strategy.endswith("Obj"):
                if len(fields) > 4 and fields[4] == "sce":
                    other.append("sce")
                else:
                    other.append("cat")
            elif strategy.endswith("Sce"):
                if "ticat" in suite_name:
                    other.append("cat")
                else:
                    other.append("sce")
            elif strategy.endswith("Cat"):
                other.append("cat")
            if "no_occ" in suite_name:
                other.append("no-occ")
            if "no-occ" in suite_name:
                other.append("no-occ")
    if fields[0] == "cxc":
        if fields[1] == "qa":
            template = "QA"
        elif fields[1] == "attr":
            template = "Obj-attr"
        elif fields[1] == "rel":
            template = "Rel-obj"
        elif fields[1] == "cap":
            template = "Cap-adj"
        elif fields[1] == "obj":
            template = "Obj-list"

        if not "wide" in suite_name:
            if "similar" in suite_name:
                strategy = "CXC-sim"
            elif "same" in suite_name:
                strategy = "CXC-same"
            elif "dissim" in suite_name:
                strategy = "CXC-dis"
        else:
            if "similar" in suite_name:
                strategy = "CXC-v-sim"
            elif "dissim" in suite_name:
                strategy = "CXC-v-dis"
        if "distant_words" in suite_name:
            strategy = "VS"
        if fields[-1] == "cap":
            other.append("+cap")
    if fields[0] == "cap":
        template = "cap-pair"
        if not "wide" in suite_name:
            if "dissim" in suite_name:
                strategy = "CXC-dis"
            elif "sim_img" in suite_name:
                strategy = "CXC-sim"
            elif "same" in suite_name:
                strategy = "CXC-same"
        else:
            if "dissim" in suite_name:
                strategy = "CXC-v-dis"
            elif "sim_img" in suite_name:
                strategy = "CXC-v-sim"
        if "cxc_sim" in suite_name:
            if "low" in suite_name:
                other.append("cxc-low")
            elif "high" in suite_name:
                other.append("cxc-high")
        elif "jacc_sim" in suite_name:
            if "low" in suite_name:
                other.append("jacc-low")
            elif "high" in suite_name:
                other.append("jacc-high")
    other = " ".join(other)
    return [template, strategy, other]
