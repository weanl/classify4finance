
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.utils import get_args
from utils.config import process_config

from data_loader.generator import DataGenerator


if __name__ == "__main__":
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    data = DataGenerator(config)
    print(data.data_agg)
    data.load()

    data.load_case_file()
    data.case_preprocess()
    case_X, case_y = data.case_load()
