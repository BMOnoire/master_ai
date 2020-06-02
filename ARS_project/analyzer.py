from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import config
import janitor as jn
import numpy as np

SHOW_ALL = False
USE_NAME_FILE = True

def cnn_analysis():
    history_path = config.general["pickle_history_path"]
    file_list = [f for f in listdir(history_path) if isfile(join(history_path, f))]
    history_list = []
    for file_name in file_list:
        correct_regex = re.match("\Ahistory_(.*)\.pickle\Z", file_name)
        if not correct_regex:
            print(file_name, " not considered")
        else:
            hist = jn.picke_load(config.general["pickle_history_path"] / file_name)
            if USE_NAME_FILE:
                new_id = correct_regex.group(1)
                hist["id"] = new_id
            history_list.append(hist)

    result_list = []
    for hist in history_list:
        id = hist["id"]
        epochs = range(1, hist["epochs"] + 1)

        if SHOW_ALL:
            plt.plot(epochs, hist["history"]["loss"], "b", label="Training Loss")
            plt.plot(epochs, hist["history"]["val_loss"], "r", label="Validation Loss")
            plt.title(str("Training and Validation losses of " + id))
            plt.legend()

            plt.figure()

            plt.plot(epochs, hist["history"]["acc"], "b", label="Training Success")
            plt.plot(epochs, hist["history"]["val_acc"], "r", label="Validation Success")
            plt.title(str("Training and Validation results of " + id))
            plt.legend()

            plt.show()

        print("Public Test ", id, " accuracy: ", hist["test"][0])
        print("Public Test ", id, " loss: ", hist["test"][1])

        result_list.append([id, hist["history"]["val_acc"], hist["history"]["val_loss"]])

    color_list = ["blue", "black", "red", "green", "pink", "magenta", "cyan", "purple", "yellow", "sienna",
                  "sandybrown", "olive", "lightslategray", "rosybrown", "dimgray", "lime"]

    plt.title(str("Accuracy results"))
    for i, res in enumerate(result_list):
        id = res[0]
        acc = res[1]
        plt.plot(epochs, acc, color_list[i], label=id)
        plt.legend()

    plt.show()

    plt.title(str("Loss results"))
    for i, res in enumerate(result_list):
        id = res[0]
        los = res[2]

        plt.plot(epochs, los, color_list[i], label=id)
        plt.legend()

    plt.show()

    # FINAL PUBLIC TEST CONFRONT
    objects = [hist["id"] for hist in history_list]
    accuracy = [hist["test"][0] for hist in history_list]
    loss = [hist["test"][1] for hist in history_list]

    plt.title("Confront test accuracy")
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, accuracy, align='center', color=color_list, alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')

    plt.show()

    plt.title("Confront test losses")
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, loss, align='center', color=color_list, alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Loss')

    plt.show()

cnn_analysis()
