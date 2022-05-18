import csv
import os
from translate_table import translate_table


def order_closeness():
    #sims = ["_sis_same", "_sis_similar", "_sis_dissim", "_distant_words"]
    #sims = ["_sis_same", "_sis_similar_wide", "_sis_similar", "_sis_dissim", "_sis_dissim_wide", "_distant_words"]
    sims = ["_sis_same", "_sis_similar", "_sis_dissim", "_sis_dissim_wide", "_distant_words"]
    #sims_obj_list = ["_same", "_similar", "_dissim", "_distant_words"]
    #sims_obj_list = ["_same", "_similar_wide", "_similar", "_dissim", "_dissim_wide", "_distant_words"]
    sims_obj_list = ["_same", "_similar", "_dissim", "_dissim_wide", "_distant_words"]
    
    order_closeness = ["ade_diff_cat_tisce", "ade_diff_sce_tisce", "ade_same_tisce"]
    order_closeness += ["ade_diff_cat_ticat", "ade_diff_sce_ticat", "ade_same_ticat"]

    order_closeness += ["ade_tfidf_same_obj_no_occ", "ade_tfidf_same_obj", "ade_tfidf_same_obj_sce_no_occ", "ade_tfidf_same_obj_sce"]
            #"same_scene_ticat", "same_scene_no_occ_ticat"]
    order_closeness += ["ade_freq_same_obj_no_occ", "ade_freq_same_obj", "ade_freq_same_obj_sce_no_occ", "ade_freq_same_obj_sce"]
    order_closeness += ["ade_tfidf_same_category_no_occ", "ade_tfidf_same_category", "ade_tfidf_same_scene_ticat", "ade_tfidf_same_scene_no_occ_ticat"]
    order_closeness += ["ade_freq_same_category_no_occ", "ade_freq_same_category", "ade_freq_same_scene_ticat", "ade_freq_same_scene_no_occ_ticat"]

    order_closeness += ["cxc_qa" + r for r in sims]
    order_closeness += ["cxc_attr" + r for r in sims]
    order_closeness += ["cxc_rel_obj" + r for r in sims]
    order_closeness += ["cxc_cap_adj" + r for r in sims]
    order_closeness += ["cxc_obj_list" + r for r in sims_obj_list]
    
    order_closeness += ["cxc_qa" + r + "_cap" for r in sims]
    order_closeness += ["cxc_attr" + r + "_cap" for r in sims]
    order_closeness += ["cxc_rel_obj" + r + "_cap" for r in sims]
    order_closeness += ["cxc_cap_adj" + r + "_cap" for r in sims]
    order_closeness += ["cxc_obj_list" + r + "_cap" for r in sims_obj_list]

    #order_closeness += ["ade_tfidf_same_obj_no_occ", "ade_tfidf_same_obj", "ade_tfidf_same_obj_sce_no_occ", "ade_tfidf_same_obj_sce"]
    #        #"same_scene_ticat", "same_scene_no_occ_ticat"]
    #order_closeness += ["ade_freq_same_obj_no_occ", "ade_freq_same_obj", "ade_freq_same_obj_sce_no_occ", "ade_freq_same_obj_sce"]
    #order_closeness += ["ade_tfidf_same_category_no_occ", "ade_tfidf_same_category", "ade_tfidf_same_scene_ticat", "ade_tfidf_same_scene_no_occ_ticat"]
    #order_closeness += ["ade_freq_same_category_no_occ", "ade_freq_same_category", "ade_freq_same_scene_ticat", "ade_freq_same_scene_no_occ_ticat"]

    img_sims = ["same_img", "sim_img", "dissim_img"]
    cap_sim_kinds = ["_cxc_sim", "_jacc_sim"]
    cap_sim_ints = ["_high", "_low"]
    wides = ["","_wide"]

    #cap_pair_closenesses = [a + b + c for (a,b,c) in zip(img_sims, cap_sim_kinds, cap_sim_ints)]
    cap_pair_closenesses = []
    for a in img_sims:
        for b in cap_sim_kinds:
            for c in cap_sim_ints:
                for d in wides:
                    #cap_pair_closenesses.append("cap_pair_"+a+b+c+d)
                    cap_pair_closenesses.append("cap_pair_"+a+d+b+c)

    order_closeness += cap_pair_closenesses
    return order_closeness


def order_all():
    #sims = ["_sis_same", "_sis_similar", "_sis_dissim", "_distant_words"]
    #sims = ["_sis_same", "_sis_similar_wide", "_sis_similar", "_sis_dissim", "_sis_dissim_wide", "_distant_words"]
    sims = ["_sis_same", "_sis_similar", "_sis_dissim", "_sis_dissim_wide", "_distant_words"]
    #sims_obj_list = ["_same", "_similar", "_dissim", "_distant_words"]
    #sims_obj_list = ["_same", "_similar_wide", "_similar", "_dissim", "_dissim_wide", "_distant_words"]
    sims_obj_list = ["_same", "_similar", "_dissim", "_dissim_wide", "_distant_words"]

    order_closeness = ["ade_diff_cat_tisce", "ade_diff_sce_tisce", "ade_same_tisce"]
    order_closeness += ["ade_diff_cat_ticat", "ade_diff_sce_ticat", "ade_same_ticat"]

    order_closeness += ["ade_tfidf_same_obj_no_occ", "ade_tfidf_same_obj", "ade_tfidf_same_obj_sce_no_occ", "ade_tfidf_same_obj_sce"]
    order_closeness += ["ade_freq_same_obj_no_occ", "ade_freq_same_obj", "ade_freq_same_obj_sce_no_occ", "ade_freq_same_obj_sce"]
    order_closeness += ["ade_tfidf_same_category_no_occ", "ade_tfidf_same_category", "ade_tfidf_same_scene_ticat", "ade_tfidf_same_scene_no_occ_ticat"]
    order_closeness += ["ade_freq_same_category_no_occ", "ade_freq_same_category", "ade_freq_same_scene_ticat", "ade_freq_same_scene_no_occ_ticat"]

    order_closeness += ["cxc_qa" + r for r in sims]
    order_closeness += ["cxc_attr" + r for r in sims]
    order_closeness += ["cxc_rel_obj" + r for r in sims]
    order_closeness += ["cxc_cap_adj" + r for r in sims]
    order_closeness += ["cxc_obj_list" + r for r in sims_obj_list]
    
    order_closeness += ["cxc_qa" + r + "_cap" for r in sims]
    order_closeness += ["cxc_attr" + r + "_cap" for r in sims]
    order_closeness += ["cxc_rel_obj" + r + "_cap" for r in sims]
    order_closeness += ["cxc_cap_adj" + r + "_cap" for r in sims]
    order_closeness += ["cxc_obj_list" + r + "_cap" for r in sims_obj_list]


    img_sims = ["same_img", "sim_img", "dissim_img"]
    cap_sim_kinds = ["_cxc_sim", "_jacc_sim"]
    cap_sim_ints = ["_high", "_low"]
    wides = ["","_wide"]

    #cap_pair_closenesses = [a + b + c for (a,b,c) in zip(img_sims, cap_sim_kinds, cap_sim_ints)]
    cap_pair_closenesses = []
    for a in img_sims:
        for b in cap_sim_kinds:
            for c in cap_sim_ints:
                for d in wides:
                    #cap_pair_closenesses.append("cap_pair_"+a+b+c+d)
                    cap_pair_closenesses.append("cap_pair_"+a+d+b+c)

    order_closeness += cap_pair_closenesses
    return order_closeness


def order_cap():
    #sims = ["_sis_similar", "_sis_dissim", "_sis_same", "_distant_words"]
    #sims_obj_list = ["_same", "_similar", "_dissim", "_distant_words"]
    #sims = ["_sis_same", "_sis_similar_wide", "_sis_similar", "_sis_dissim", "_sis_dissim_wide", "_distant_words"]
    sims = ["_sis_same", "_sis_similar", "_sis_dissim", "_sis_dissim_wide", "_distant_words"]
    #sims_obj_list = ["_same", "_similar_wide", "_similar", "_dissim", "_dissim_wide", "_distant_words"]
    sims_obj_list = ["_same", "_similar", "_dissim", "_dissim_wide", "_distant_words"]

    #order_closeness = ["ade_diff_cat", "ade_diff_sce", "ade_same"]
    order_closeness1 = ["cxc_qa" + r for r in sims]
    order_closeness1 += ["cxc_attr" + r for r in sims]
    order_closeness1 += ["cxc_rel_obj" + r for r in sims]
    order_closeness1 += ["cxc_cap_adj" + r for r in sims]
    order_closeness1 += ["cxc_obj_list" + r for r in sims_obj_list]

    order_closeness2 = ["cxc_qa" + r + "_cap" for r in sims]
    order_closeness2 += ["cxc_attr" + r + "_cap" for r in sims]
    order_closeness2 += ["cxc_rel_obj" + r + "_cap" for r in sims]
    order_closeness2 += ["cxc_cap_adj" + r + "_cap" for r in sims]
    order_closeness2 += ["cxc_obj_list" + r + "_cap" for r in sims_obj_list]

    order_closeness = []
    for (a,b) in zip(order_closeness1, order_closeness2):
        order_closeness.append(a)
        order_closeness.append(b)

    return order_closeness

def order_freq_tfidf():
    base = ["same_obj", "same_obj_no_occ", "same_category", "same_category_no_occ",
            "same_obj_sce", "same_obj_sce_no_occ", "same_scene", "same_scene_no_occ",
            "same_scene_ticat", "same_scene_no_occ_ticat"]
    order = []
    for b in base:
        for add in ["tfidf", "freq"]:
            order.append("ade_" + add + "_" + b)
    return order


if __name__=="__main__":
    #data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap_binary"
    data_base_path = "/home/jseltmann/data/results_coco_val_binary_addition"
    all_models_dict = dict()
    order = []
    for model in ["bert", "visual-bert", "w2v"]:
        line_dict = dict()
        csv_path = os.path.join(data_base_path, model+".csv")
        #print(csv_path)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for line in reader:
                if line[0] == "model":
                    continue
                line_new = []
                for i, entry in enumerate(line):
                    if i > 1 and len(entry)>0:
                        entry = round(float(entry), 2)
                    line_new.append(entry)
                line = line_new
                line_dict[line[0].strip()] = line
                if not line[0].strip() in order:
                    order.append(line[0].strip())
        for line_name in order:
            if not line_name in all_models_dict:
                all_models_dict[line_name] = dict()
            line = line_dict[line_name]
            #all_models_dict[line_name][model] = [line_name, model] + [line[1]]
            all_models_dict[line_name][model] = round(float(line[1]), 2)

    order_orig = order

    #data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap"
    data_base_path = "/home/jseltmann/data/results_coco_val_sg"
    sg_models_dict = dict()
    order = []
    for model in ["bert", "visual-bert", "w2v"]:
        line_dict = dict()
        if model in ["bert", "lxmert", "visual-bert"]:
            csv_path = os.path.join(data_base_path, model+"-base_sg.csv")
        else:
            csv_path = os.path.join(data_base_path, model+"_sg.csv")
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for line in reader:
                if line[0] == "model":
                    continue
                line_new = []
                for i, entry in enumerate(line):
                    if i > 1 and len(entry)>0:
                        entry = round(float(entry), 2)
                    line_new.append(entry)
                line = line_new
                line_dict[line[0].strip()] = line
                if not line[0].strip() in order:
                    order.append(line[0].strip())
        for line_name in order:
            if not line_name in sg_models_dict:
                sg_models_dict[line_name] = dict()
            line = line_dict[line_name]
            #sg_models_dict[line_name][model] = [line_name, model] + [line[1]]
            sg_models_dict[line_name][model] = round(float(line[1]), 2)
        


    #order = order_cap()
    #with open("acc_by_cap.csv", "w", newline='') as tablef:
    #    writer = csv.writer(tablef, delimiter="|")
    #    writer.writerow(["model", "suite", "binary", "sg"])
    #    #for model in ["bert", "lxmert", "w2v"]:
    #    for model in ["bert", "visual-bert", "w2v"]:
    #        for suite_name in order:
    #            #line = all_models_dict[suite_name][model]
    #            if not suite_name in all_models_dict:
    #                #print(suite_name)
    #                continue
    #            binary = all_models_dict[suite_name][model]
    #            try:
    #                sg = sg_models_dict[suite_name][model]
    #            except Exception as e:
    #                sg = None
    #            line = [model, suite_name, binary, sg]
    #            writer.writerow(line)



    #order = order_freq_tfidf()
    #with open("acc_by_freq_tfidf.csv", "w", newline='') as tablef:
    #    writer = csv.writer(tablef, delimiter="|")
    #    writer.writerow(["model", "suite", "binary", "sg"])
    #    #for model in ["bert", "lxmert", "w2v"]:
    #    for model in ["bert", "visual-bert", "w2v"]:
    #        for suite_name in order:
    #            #line = all_models_dict[suite_name][model]
    #            if not suite_name in all_models_dict:
    #                #print(suite_name)
    #                continue
    #            binary = all_models_dict[suite_name][model]
    #            try:
    #                sg = sg_models_dict[suite_name][model]
    #            except Exception as e:
    #                sg = None
    #            line = [model, suite_name, binary, sg]
    #            writer.writerow(line)


    order = order_closeness()
    with open("results_all.csv", "w") as tablef:
        #for model in ["bert", "lxmert", "w2v"]:
        for suite_name in order:
            if not suite_name in all_models_dict:
                continue
            template, strategy, other = translate_table(suite_name, 0)
            line = template + " & " + strategy + " & " + other
            for model in ["bert", "visual-bert", "w2v"]:
                binary = round(all_models_dict[suite_name][model],2)
                line += " & " + str(binary)
            for model in ["bert", "visual-bert", "w2v"]:
                try:
                    sg = round(sg_models_dict[suite_name][model],2)
                except Exception as e:
                    sg = None
                line += " & " + str(sg)
            for model in ["bert", "visual-bert", "w2v"]:
                #line = all_models_dict[suite_name][model]
                binary = all_models_dict[suite_name][model]
                try:
                    sg = sg_models_dict[suite_name][model]
                    diff = round(binary - sg, 3)
                except Exception as e:
                    sg = None
                    diff = None
                line += " & " + str(diff)
                #line = [model, suite_name, binary, sg]
                #writer.writerow(line)
            line += "\\\\\n"
            tablef.write(line)
            tablef.write("\\hline\n")

   # order = order_orig
   # with open("diff_binary_sg.csv", "w", newline='') as tablef:
   #     writer = csv.writer(tablef, delimiter="|")
   #     writer.writerow(["model", "suite", "binary", "sg", "diff"])
   #     #for model in ["bert", "lxmert", "w2v"]:
   #     lines = []
   #     for suite_name in order:
   #         if not suite_name in all_models_dict:
   #             #print(suite_name)
   #             continue
   #         #for model in ["bert", "visual-bert", "w2v"]:
   #         for model in ["bert", "visual-bert"]:
   #             #line = all_models_dict[suite_name][model]
   #             binary = all_models_dict[suite_name][model]
   #             try:
   #                 sg = sg_models_dict[suite_name][model]
   #             except Exception as e:
   #                 sg = None
   #             if sg is not None:
   #                 diff = round(binary - sg,3)
   #             else:
   #                 diff = -999999999999999999999999999999
   #             template, strategy, other = translate_table(suite_name, 0)
   #             line = [model, template, strategy, other, binary, sg, diff]
   #             lines.append(line)
   #     lines = sorted(lines, key=lambda l: l[-1], reverse=True)
   #     for line in lines:
   #         writer.writerow(line)
