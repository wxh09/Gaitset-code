from model.initialization import initialization
from config import conf
import argparse
import warnings

warnings.filterwarnings('ignore')

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Train')
# parser.add_argument('--cache', default=True, type=boolean_string,help='cache: if set as TRUE all the training data will be loaded at once before the training start. Default: TRUE')
parser.add_argument('--train', default=True, type=boolean_string,help='train:true. Default: TRUE')
opt = parser.parse_args()

m = initialization(conf, train=opt.train)[0]

print("Training START")
m.fit()
print("Training COMPLETE")
