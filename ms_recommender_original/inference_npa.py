import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device_no", type=str)
    parser.add_argument("--infer_model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--news_file", type=str)
    parser.add_argument("--behaviors_file", type=str)
    return parser.parse_args()


args = parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_no
import time
import pickle
import sys
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # only show error messages

from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.npa import NPAModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

epochs = args.epochs
seed = 40
batch_size = 32
data_dir = args.data_dir

### Download and load data
news_file = args.news_file
behaviors_file = args.behaviors_file
wordEmb_file = os.path.join(data_dir, "MINDlarge_utils", "embedding.npy")
userDict_file = os.path.join(data_dir, "MINDlarge_utils", "uid2index.pkl")
wordDict_file = os.path.join(data_dir, "MINDlarge_utils", "word_dict.pkl")
yaml_file = os.path.join(data_dir, "MINDlarge_utils", r'npa.yaml')

### Create hyper-parameters
hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs)
print(hparams)

iterator = MINDIterator

model = NPAModel(hparams, iterator, seed=seed)
model.model.load_weights(args.infer_model_path)
print(f'load {args.infer_model_path} successfully !')

behaviors_suffix = behaviors_file.split("/")[-1].split(".")[0]
print('suffix', behaviors_suffix)

os.makedirs(args.save_dir, exist_ok=True)

start_time = time.time()
idx_list, labels_list, predictions_list = model.run_slow_eval(
    news_file, behaviors_file
)
idx_path = os.path.join(args.save_dir, f'{behaviors_suffix}_idx.pkl')
predictions_path = os.path.join(args.save_dir, f'{behaviors_suffix}_predictions.pkl')
with open(idx_path, "wb") as f:
    pickle.dump(idx_list, f)
with open(predictions_path, "wb") as f:
    pickle.dump(predictions_list, f)
print(f'it costs {time.time() - start_time} s to infer the results')
