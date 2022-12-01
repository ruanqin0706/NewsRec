import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device_no", type=str)
    parser.add_argument("--infer_model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--news_file", type=str)
    parser.add_argument("--behaviors_file", type=str)
    parser.add_argument("--gen_news", action='store_true')
    return parser.parse_args()


args = parse_args()

import sys
import os
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # only show error messages
import pickle
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

epochs = args.epochs
seed = 40
batch_size = 32
data_dir = args.data_dir

# Download and load data
# valid_news_file = os.path.join(data_dir, 'test', r'news.tsv')
# valid_behaviors_file = os.path.join(data_dir, 'test', r'behaviors_new.tsv')
news_file = args.news_file
behaviors_file = args.behaviors_file
wordEmb_file = os.path.join(data_dir, "MINDlarge_utils", "embedding.npy")
userDict_file = os.path.join(data_dir, "MINDlarge_utils", "uid2index.pkl")
wordDict_file = os.path.join(data_dir, "MINDlarge_utils", "word_dict.pkl")
yaml_file = os.path.join(data_dir, "MINDlarge_utils", r'nrms.yaml')

# Create hyper-parameters
hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)

iterator = MINDIterator

# Train the LSTUR model
model = NRMSModel(hparams, iterator, seed=seed)
model.model.load_weights(args.infer_model_path)
print(f'load {args.infer_model_path} successfully !')

behaviors_suffix = behaviors_file.split("/")[-1].split(".")[0]
print('suffix', behaviors_suffix)

os.makedirs(args.save_dir, exist_ok=True)

if args.gen_news:
    start_time = time.time()
    news_vecs = model.run_news(news_file)
    news_path = os.path.join(args.save_dir, f'{behaviors_suffix}_news.pkl')
    with open(news_path, "wb") as f:
        pickle.dump(news_vecs, f)
    print(f'it costs {time.time() - start_time} s to generate news vecs')
model.run_fast_eval()
start_time = time.time()
user_vecs = model.run_user(news_file, behaviors_file)
user_path = os.path.join(args.save_dir, f'{behaviors_suffix}_user.pkl')
with open(user_path, "wb") as f:
    pickle.dump(user_vecs, f)
print(f'it costs {time.time() - start_time} s to generate users vecs')
