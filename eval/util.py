from sklearn.metrics import recall_score, precision_score, f1_score


def print_information(real_values, predictions, filename):
    with open(filename, 'w') as f:
        print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
        f.write("Weighted Recall {} \n".format(recall_score(real_values, predictions, average='weighted')))
        print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
        f.write("Weighted Precision {}\n".format(precision_score(real_values, predictions, average='weighted')))
        print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))
        f.write("Weighted F1 Score {}\n".format(f1_score(real_values, predictions, average='weighted')))

        print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))
        f.write("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))
