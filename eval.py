# this file is mostly copied from this project: https://github.com/AnneBeyer/coherencegym

import configparser
import argparse
import os
import json
import numpy as np
import csv
import re
import pandas as pd
from tqdm import tqdm
import math

#from bokeh.plotting import figure, output_file, show
#from bokeh.models import ColumnDataSource, ranges, LabelSet, CDSView, BooleanFilter
#from bokeh.resources import CDN
#from bokeh.embed import autoload_static

import lm_zoo
import syntaxgym as sg


def _get_model_names(args, config):
    if args["models"] is not None:
        models = args["models"]
    else:
        models = config["Models"]["models"].split("\n")
    return models


def _get_suite_paths(args, config):
    suites_to_use = []
    suites_path = config["Suites"]["suitespath"]
    if args["suites"]:
        for suite_name in args["suites"]:
            combined_path = os.path.join(suites_path, suite_name)
            if "/" in suite_name:
                if os.path.exists(combined_path):
                    suites_to_use.append(combined_path)
                else:
                    suites_to_use.append(suite_name)
            else:
                sub_suites = os.listdir(combined_path)
                for s in sub_suites:
                    sub_path = os.path.join(combined_path, s)
                    suites_to_use.append(sub_path)
    else:
        # use all suites
        suites_dirs = os.listdir(suites_path)
        for sd in suites_dirs:
            dp = os.path.join(suites_path, sd)
            ind_suites = os.listdir(dp)
            for ind_suite in ind_suites:
                ind_suite_path = os.path.join(dp, ind_suite)
                suites_to_use.append(ind_suite_path)
    return suites_to_use


def _item_scores(item, condition_names, region_index):
    """
    Get scores for the two conditions in a suite item.
    """
    scores = dict()
    for cond in item["conditions"]:
        name = cond["condition_name"]
        region = cond["regions"][region_index]
        if "sum" in region["metric_value"].keys():
            scores[name] = region["metric_value"]["sum"]
        else:
            scores[name] = region["metric_value"]["mean"]
    return scores


def _calculate_averages(results):
    """
    Calculate various values to sum up the quality of the results.
    Returns a dict where the results are already formatted strings
    for writing to the csv.
    """

    with_surprisals, eval_results = results

    avgs = dict()
    avgs["All"] = dict()
    avgs["All"][" items "] = " All   "
    avgs["True"] = dict()
    avgs["True"][" items "] = " True  "
    avgs["True"][" % Corr "] = "{: 4.3f}  ".format(1)
    avgs["False"] = dict()
    avgs["False"][" items "] = " False "
    avgs["False"][" % Corr "] = "{: 4.3f}  ".format(0)

    true_false = eval_results["result"].to_list()
    corr = [b for b in true_false if b == True]
    perc_corr = len(corr) / len(true_false)
    avgs["All"][" % Corr "] = "{: 4.3f}  ".format(perc_corr)

    cns = with_surprisals.condition_names
    scores = dict()
    for cn in cns:
        scores[cn] = []
    
    formula = with_surprisals.as_dict()["predictions"][0]["formula"]
    match = re.search("[0-9]+;", formula)
    if match:
        region_index = int(match.group(0)[:-1]) - 1
    else:
        region_index = -1

    for item in with_surprisals.items:
        curr_scores = _item_scores(item, cns, region_index)
        for cn in curr_scores:
            scores[cn].append(curr_scores[cn])

    for cn in scores:
        scores[cn] = np.array(scores[cn])
        avgs["All"][" " + cn + " mean "] = "{: 8.3f}    ".format(np.mean(scores[cn]))
        avgs["All"][" " + cn + " std "] = "{: 8.3f}    ".format(np.std(scores[cn]))
        scores_true = np.array([s for (s,b) in zip(scores[cn], true_false) if b == True])
        scores_false = np.array([s for (s,b) in zip(scores[cn], true_false) if b == False])
        avgs["True"][" " + cn + " mean "] = "{: 8.3f}    ".format(np.mean(scores_true))
        avgs["True"][" " + cn + " std "] = "{: 8.3f}    ".format(np.std(scores_true))
        avgs["False"][" " + cn + " mean "] = "{: 8.3f}    ".format(np.mean(scores_false))
        avgs["False"][" " + cn + " std "] = "{: 8.3f}    ".format(np.std(scores_false))

    headers = []
    for cn in scores:
        headers.append(" " + cn + " mean ")
        headers.append(" " + cn + " std ")

    return avgs, headers


def write_results_curr_model(results, out_dir, model_name, model_name_length):
    #out_dir = args["output_directory"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for suite in results:
        suite_json_dir = os.path.join(out_dir, suite)
        if not os.path.exists(suite_json_dir):
            os.makedirs(suite_json_dir)

        json_out_path = os.path.join(suite_json_dir, model_name+".json")
        with open(json_out_path, "w") as out_file:
            with_surprisals, eval_results = results[suite]
            eval_results = eval_results["result"].to_list()
            with_surprisals = with_surprisals.as_dict()
            to_dump = {"raw": with_surprisals, "results": eval_results}
            json.dump(to_dump, out_file)
        curr_avgs, headers = _calculate_averages(results[suite])
        model_key = "Model" + " " * (model_name_length - 4)
        curr_avgs["All"][model_key] = model_name + " " * (model_name_length - len(model_name) + 1)
        curr_avgs["True"][model_key] = model_name + " " * (model_name_length - len(model_name) + 1)
        curr_avgs["False"][model_key] = model_name + " " * (model_name_length - len(model_name) + 1)

        suite_csv_path = os.path.join(out_dir, suite+".csv")
        if os.path.exists(suite_csv_path):
            new_file = False
        else:
            new_file = True

        with open(suite_csv_path, "a+", newline='') as csv_file:
            fieldnames = ["Model" + " " * (model_name_length - 4), " items ", " % Corr "] + headers
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter="|")
            if new_file:
                writer.writeheader()

            for category in curr_avgs:
                writer.writerow(curr_avgs[category])


def get_visualizations_multiple(csv_paths, out_path=None):
    """
    Create visualizations for a number of files.
    """

    tag_data = dict()

    for csv_path in csv_paths:
        suite_name, suite_data = get_visualization_single(csv_path, out_path)
        tag_data[suite_name] = suite_data

    if out_path:
        with open(out_path+"/tags.json", "w") as out_file:
            json.dump(tag_data, out_file)
        


def get_visualization_single(csv_path, out_path=None):
    """
    Create visualization out of a csv file.
    Return a tag to be included in html.
    """

    df = pd.read_csv(csv_path, sep="\s+\|\s+")
    all_items = df.loc[df["items"] == "All"]
    true_items = df.loc[df["items"] == "True"]
    false_items = df.loc[df["items"] == "False"]

    tags = []

    suite_name = csv_path.split("/")[-1][:-4]
    if out_path:
        suite_path = os.path.join(out_path, suite_name)

    suite_data = dict()

    for name, items in zip(["All","True","False"], [all_items, true_items, false_items]):
        plot_width = 400
        num_bars = len(df)
        width = plot_width / num_bars
        x = [n * width + (width/2) for n in range(num_bars)]
        for col_name in items.columns.values:
            if col_name == "Model" or col_name == "items":
                continue
            if not col_name in suite_data:
                suite_data[col_name] = dict()
            suite_data[col_name][name] = dict()

            source = ColumnDataSource(dict(x=items["Model"], y=items[col_name]))
            # ignore NaN entries
            booleans = [True if not math.isnan(y_val) else False for y_val in source.data['y']]
            view = CDSView(source=source, filters=[BooleanFilter(booleans)])

            max_val = max([v for v in items[col_name] if not math.isnan(v)])
            max_val = max_val + max_val * 0.1
            p = figure(plot_width=plot_width, plot_height=400,
                       x_axis_label = "Models", y_axis_label = name + " results " + col_name,
                       x_range = source.data["x"], y_range = ranges.Range1d(start=0,end=max_val))
            labels = LabelSet(x='x', y='y', text = 'y', level='glyph', x_offset=-13.5, y_offset=0, source=source, render_mode='canvas')
            p.vbar(source=source, x='x', top='y', bottom=0, width=0.3, view=view)
            p.add_layout(labels)

            if out_path:
                col_name_new = col_name.replace(" ", "_").replace("%", "")
                js, tag = autoload_static(p, CDN, suite_name + "/" + name + col_name_new)
                src = tag.split('"')[1]
                suite_data[col_name][name]["src"] = src
                tag_id = tag.split('"')[3]
                suite_data[col_name][name]["id"] = tag_id
                if not os.path.exists(suite_path):
                    os.makedirs(suite_path)
                with open(suite_path + "/" + name + col_name_new, "w") as out_file:
                    out_file.write(js)

                tags.append(tag)

            if not out_path:
                show(p)

    return suite_name, suite_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run syntaxgym with various dialog models and test suites. You have to call it with sudo, since each model is a docker container.")
    subparsers = parser.add_subparsers(dest="action")

    # parser for run
    run_parser = subparsers.add_parser("run", help="Run models on test suites.")
    #run_parser.add_argument("output_directory", help="Directory to save output to. Saves syntaxgym output to a json file and creates one csv file for the results on each suite.")
    #run_parser.add_argument("--suites", "-s", nargs="*", help="Suites to use. Use all suites if not given. Suites must all be stored in one directory, with one subdirectory for each type of suite. Either give subdirectory name to run on all suites of the type or `subdirectory/suite` to run on one particular suites.")
    #run_parser.add_argument("--models", "-m", nargs="*", help="Models to use.")
    #run_parser.add_argument("--config", "-c", default="config.ini", help="Path to config file. Config file should contain path to suites and list of available models.")

    # parser for show
    show_parser = subparsers.add_parser("show", help="Show available suites or models.")
    show_parser.add_argument("--config", "-c", default="config.ini", help="Path to config file. Config file should contain path to suites and list of available models.")
    show_parser.set_defaults(subparser_name="show")
    show_parser.add_argument("category", choices=["models", "suites"], help="`models` shows models. `suites` shows suite types.")
    show_parser.add_argument("suite-type", default=None, nargs='?', help="If given, show suites belonging to the given suite type.")

    # parser for visualize
    vis_parser = subparsers.add_parser("visualize", help="Generate <script> tag to include graph in html.")
    vis_parser.add_argument("infiles", nargs="*")
    vis_parser.add_argument("--config", "-c", default="config.ini", help="Path to config file. Config file should contain path to suites and list of available models.")
    vis_parser.add_argument("--outfile", "-o", required=False)

    args = vars(parser.parse_args())
    #config = configparser.ConfigParser()
    #config.read(args["config"])

    if args["action"] == "run":
        model_names = ["bert-base", "lxmert-base"]
        results_path = "/home/jseltmann/data/results_debug"

        results = dict()
        model_name_length = max([len(mn) for mn in model_names])
        for model_name in model_names:
            results = dict()
            print(model_name)
            model = lm_zoo.get_registry()[model_name]
            with open("generate_suites/generation_combinations_small.csv", newline='') as gcf:
                not_comment = lambda line: line[0]!='#'
                reader = csv.reader(filter(not_comment, gcf), delimiter=",")
                for i, row in enumerate(reader):
                    try:
                        if i == 0:
                            continue
                        print("\t" + row[4])
                        suites_dir = os.path.join("/home/jseltmann/data/suites", row[3])
                        suite_path = os.path.join(suites_dir, row[4]+"_test.json")

                        suite_name = row[3] + "_" + row[4]
                        if not suite_name in results:
                            results[suite_name] = dict()
                        with_surprisals = sg.compute_surprisals(model, suite_path)
                        eval_results = sg.evaluate(with_surprisals)
                        results[suite_name] = with_surprisals, eval_results
                    except Exception as e:
                        print(e)
                write_results_curr_model(results, results_path, model_name, model_name_length)

    elif args["action"] == "show":
        print("not implemented right now")
        #if args["category"] == "suites":
        #    suites_path = config["Suites"]["suitespath"]
        #    if args["suite-type"]:
        #        suite_dir = args["suite-type"]
        #        path = os.path.join(suites_path, suite_dir)
        #        for suite in os.listdir(path):
        #            print(os.path.join(suite_dir, suite))
        #    else:
        #        suites_dirs = os.listdir(suites_path)
        #        for d in suites_dirs:
        #            print(d)

        #elif args["category"] == "models":
        #    models = config["Models"]["models"].split("\n")
        #    for model in models:
        #        print(model)
    elif args["action"] == "visualize":
        print("not implemented right now")
        #get_visualizations_multiple(args["infiles"], args["outfile"])
