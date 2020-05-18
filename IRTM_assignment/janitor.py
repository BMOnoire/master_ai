import os
import pickle


def create_dir(path):
    path = str(path)
    if os.path.exists(path):
        print("Directory %s exists" % path)
        return True
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        return False
    else:
        print("Successfully created the directory %s " % path)
        return True


def pickle_load(pickle_path):
    if pickle_path.exists():
        pickle_in = open(str(pickle_path), "rb")
        return pickle.load(pickle_in)
    else:
        return None


def pickle_save(object, pickle_path):
    pickle_out = open(str(pickle_path), "wb")
    pickle.dump(object, pickle_out)
    pickle_out.close()
