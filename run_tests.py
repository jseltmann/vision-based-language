from lm_zoo import get_registry
from syntaxgym import compute_surprisals, evaluate
import csv
import os
import logging
import json

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename='errors.log')

    models = ["bert-base"]
    for model_name in models:
        model = get_registry()[model_name]
        with open("generate_suites/generation_combinations.csv", newline='') as gcf:
            not_comment = lambda line: line[0]!='#'
            reader = csv.reader(filter(not_comment, gcf), delimiter=",")

            for i, row in enumerate(reader):
                try:
                    if i == 0:
                        continue
                    print(row[4])
                    suites_dir = os.path.join("/home/jseltmann/data/suites/", row[3])
                    suite_path = os.path.join(suites_dir, row[4]+"_test.json")

                    results_dir = os.path.join("/home/jseltmann/data/results/", row[3])
                    results_speci_dir = os.path.join(results_dir, row[4])
                    if not os.path.exists(results_speci_dir):
                        os.makedirs(results_speci_dir)
                    results_path = os.path.join(results_speci_dir, model_name+".csv")
                    if os.path.exists(results_path):
                        continue
                    surp_path = os.path.join(results_speci_dir, model_name+".json")

                    suite = compute_surprisals(model, suite_path)
                    with open(surp_path, "w") as surp_file:
                        json.dump(suite.as_dict(), surp_file)
                    results = evaluate(suite)
                    results.to_csv(results_path)
                except:
                    logging.exception(model_name + " " + row[4])


