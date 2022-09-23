from visualizer import get_local
get_local.activate()
import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
from torch.utils.data import DataLoader
from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
from trainer import Trainer
# from trainer_reformer import Trainer
# from trainer_lstm import Trainer
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

def grid_show(to_shows, n_layer, ant, lenf, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    # print('rows =', rows)
    fig, axs = plt.subplots(rows, cols)#, figsize=(rows*8.5, cols*2)
    # print('axs =', axs)
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    # plt.show()
    plt.savefig('vis/' + str(lenf) + '_ant' + str(ant) + '_layer' + str(n_layer) + '-focal*crloss_attweight.png')

def visualize_head(att_map, ant, lenf, head):
    # ax = plt.gca()
    # # Plot the heatmap
    # im = ax.imshow(att_map)#imshow
    # # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    # # plt.show()
    # plt.savefig('vis/all_xgb_10000epoch_lr0.0001_w.png')
    summary = pd.DataFrame(np.average(att_map, axis=0))  #
    summary.to_csv('results/' + str(lenf) + '_ant' + str(ant) + '_layer2_head' + str(head) + '_plot-focal*crloss_attweight.csv', sep='\t')
    plt.figure(figsize=(30, 10))
    plt.plot(range(0, lenf), np.average(att_map, axis=0))
    plt.savefig('vis/' + str(lenf) + '_ant' + str(ant) + '_layer2_head' + str(head) + '_plot-focal*crloss_attweight.png')

def visualize_heads(att_map, n_layer, ant, lenf, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, n_layer, ant, lenf, cols=cols)

def main(args):
    print(args)
    # Load tokenizer
    if args.tokenizer == 'sentencepiece':
        tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
    else:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
        tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file=args.vocab_file)
    result_summary = []
    results = []
    stride = 1000
    for i in range(0, 2):#182098 // stride 1736 // stride +
        # Build DataLoader
        start = i * stride
        if i == 1736 // stride:
            end = 1736
        else:
            end = start + stride
        train_dataset, test_dataset, all_labels, weights = create_examples(args, tokenizer, start, end, i)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # Build Trainer
        trainer = Trainer(args, train_loader, test_loader, tokenizer, all_labels, weights)

        # Train & Validate
        h_accs_t = 0.0
        h_epoch = 0
        h_accs0 = 0
        h_row = 0.0
        h_ave = 0.0
        h_lr = 0
        h_loss = 0.0
        # h_accs_t_1 = 0.0
        # h_epoch_1 = 0
        # h_accs0_1 = 0

        for epoch in range(1, args.epochs+1):
            trainer.train(epoch)
            lr, loss, accs0, accs_t, ave_row, ave = trainer.validate(epoch)
            if ave_row > h_row:
                h_accs_t = accs_t
                h_epoch = epoch
                h_accs0 = accs0
                h_row = ave_row
                h_ave = ave
                h_lr = lr
                h_loss = loss
            trainer.save(epoch, args.output_model_prefix)
            results.append([epoch, lr, loss, accs0, accs_t, ave_row, ave])
        result_summary.append([h_epoch, h_lr, h_loss, h_accs0, h_accs_t, h_row, h_ave])
        result_csv = pd.DataFrame(results, columns=['epoch', 'lr', 'loss ', 'raw acc', 'acc', 'ave_row', 'ave'])
        result_csv.to_csv('results/' + str(args.max_seq_len) + '_attweight_ant' + str(i) + '_results-focal*crloss.csv', sep='\t')
    summary = pd.DataFrame(result_summary, columns=['epoch', 'lr', 'loss ', 'raw acc', 'acc', 'ave_row', 'ave'])
    summary.to_csv('results/' + str(args.max_seq_len) + '_attweight_ant' + str(i) + '-focal*crloss.csv', sep='\t')
        # summary = pd.DataFrame(result_summary, columns=['epoch', 'lr', 'loss ', 'raw acc', 'acc', 'ave_row', 'ave'])
        # summary.to_csv('results/' + str(i) + '2000_acc.csv', sep='\t')
    cache = get_local.cache
    # print(list(cache.keys()))
    attention_maps = cache['MultiHeadAttention.forward']
    # print(len(attention_maps))
    # print(attention_maps[0])
    print(np.sum(attention_maps[2], axis=0).shape)
    # print(np.sum(attention_maps[2], axis=0))
    # print(np.average(attention_maps[2], axis=0).shape)
    # np.average(attention_maps[2], axis=0)
    # print(attention_maps[0][0, 1].shape)

    visualize_head(np.average(attention_maps[2], axis=0)[0], i, args.max_seq_len, 0)
    visualize_head(np.average(attention_maps[2], axis=0)[1], i, args.max_seq_len, 1)
    visualize_head(np.average(np.average(attention_maps[2], axis=0), axis=0), i, args.max_seq_len, 2)
    for j in range(len(attention_maps)):
        visualize_heads(np.average(attention_maps[j], axis=0), j, i, args.max_seq_len, cols=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset',             default='../data/mic_5000.csv',           type=str, help='dataset')
    parser.add_argument('--vocab_file',          default='../vocab/vocab_g_1.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--tokenizer',           default='nltk_moses',  type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER_CLASSES.keys()))#
    parser.add_argument('--pretrained_model',    default='wiki.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--output_model_prefix', default='model',          type=str, help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size',     default=32,   type=int,   help='batch size')
    parser.add_argument('--max_seq_len',    default=2000,  type=int,   help='the maximum size of the input sequence')
    # Train parameters
    parser.add_argument('--epochs',         default=1200,   type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda',        action='store_true')#default=True action='store_true'
    # Model parameters
    parser.add_argument('--hidden',         default=32,  type=int,   help='the number of expected features in the transformer')
    parser.add_argument('--n_layers',       default=3,    type=int,   help='the number of layers in the multi-head attention network')
    parser.add_argument('--n_attn_heads',   default=2,    type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--dropout',        default=0.2,  type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden',     default=512, type=int,   help='the dimension of the feedforward network')
    
    args = parser.parse_args()
    
    main(args)