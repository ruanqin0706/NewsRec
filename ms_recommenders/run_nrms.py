import argparse
import sys
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # only show error messages

from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int)
    return parser.parse_args()


args = parse_args()

epochs = args.epochs
seed = args.seed
batch_size = args.batch_size
data_dir = args.data_dir

# Download and load data
train_news_file = os.path.join(data_dir, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_dir, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_dir, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_dir, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_dir, "utils", "embedding.npy")
userDict_file = os.path.join(data_dir, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_dir, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_dir, "utils", r'nrms.yaml')


# Create hyper-parameters
hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)


os.makedirs(args.model_dir, exist_ok=True)

