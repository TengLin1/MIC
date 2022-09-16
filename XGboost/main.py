import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
import matplotlib
matplotlib.use('agg')

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}


def main(args):
    print(args)
    for i in range(0, 14):#182098 // stride 1736 // stride +
        create_examples(args, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', default='../data/mic_5000_1.csv', type=str, help='dataset') # for cluster
    parser.add_argument('--dataset', default='../data/mic_5000.csv', type=str, help='dataset')
    parser.add_argument('--vocab_file', default='../vocab/vocab_g_1.vocab', type=str, help='vocabulary path')
    parser.add_argument('--tokenizer', default='nltk_moses', type=str,
                        help='tokenizer to tokenize input corpus. available: sentencepiece, ' + ', '.join(
                            TOKENIZER_CLASSES.keys()))  #
    parser.add_argument('--pretrained_model', default='wiki.model', type=str,
                        help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--output_model_prefix', default='model', type=str, help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--max_seq_len', default=2000, type=int, help='the maximum size of the input sequence')
    # Train parameters
    parser.add_argument('--epochs', default=3, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', default=True)  # default=True action='store_true'
    # Model parameters
    parser.add_argument('--hidden', default=32, type=int, help='the number of expected features in the transformer')
    parser.add_argument('--n_layers', default=3, type=int,
                        help='the number of layers in the multi-head attention network')
    parser.add_argument('--n_attn_heads', default=2, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--dropout', default=0.2, type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden', default=512, type=int, help='the dimension of the feedforward network')
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--task_name", default=None, type=str,
                        help="The name of the task to train selected in the list: ")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='albert_base_v2/', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--spm_model_file", default='', type=str)
    args = parser.parse_args()
    
    main(args)