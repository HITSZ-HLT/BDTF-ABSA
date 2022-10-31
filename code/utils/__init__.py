import numpy as np
import json
import os



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)


def append_json(file_name, obj, mode='a'):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode=mode, encoding='utf-8') as f:
        if type(obj) is dict:
            string = json.dumps(obj)
        elif type(obj) is list:
            string = ' '.join([str(item) for item in obj])
        elif type(obj) is str:
            string = obj
        else:
            raise Exception()

        string = string + '\n'
        f.write(string)

    
def mkdir_if_not_exist(path):
    dir_name, file_name = os.path.split(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_json(json_obj, file_name):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='w+', encoding='utf-8-sig') as f:
        json.dump(json_obj, f, indent=4, cls=NpEncoder)



def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()



def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)


def yield_data_file(data_dir):
    for file_name in os.listdir(data_dir):
        yield os.path.join(data_dir, file_name)
