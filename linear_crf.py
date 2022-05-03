import torch.nn as nn
import torch
from typing import Dict, List
from typing import Tuple
import pickle
import numpy as np
from dynamic_lstm import dynamicLSTM
from dynamic_rnn import dynamicRNN
from dynamic_gru import dynamicGRU
from self_defined_lstm_linear import LSTM
from self_defined_gru_linear import GRU
from self_defined_rnn_linear import RNN

# from self_defined_simple_recurrence_VA_EW import RNN as VA_EW
# from self_defined_simple_recurrence_MVMA_ME import RNN as MVMA_ME
# from matrix_multi_model import MatMultiModel

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    output the score of all labels.
    """

    def __init__(self, label_size: int, input_dim:int,
                 hidden_dim: int,
                 drop_lstm:float=0.5,
                 num_lstm_layers: int =1):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = label_size
        print("[Model Info] Input size to LSTM: {}".format(input_dim))
        print("[Model Info] LSTM Hidden Size: {}".format(hidden_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)
        self.hidden2tag = nn.Linear(hidden_dim, self.label_size)

    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)

        outputs = self.hidden2tag(feature_out)
        return outputs[recover_idx]
    
class VAWEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        '''
        Average the GRU hidden vectors
        '''
        super(VAWEncoder, self).__init__()
        #self.embeddings.weight.data.uniform_(-be, be)
        #self.cnn = nn.Conv1d(embed_dim, hidden_dim, kernel_size=20, padding=0, stride=20,bias=False)
        self.max_step = 20
        self.cnn = nn.Parameter(torch.randn(self.max_step, embed_dim, hidden_dim).uniform_(-0.1, 0.1))
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(embed_dim, hidden_dim)
        print('VA-W model')

    # batch_size * sent_l * dim
    def forward(self, word_rep, seq_lengths=None):
        '''
        Args:
            word_rep: word embeddings, batch_size*max_len*emb_dim, Long Tensor
            seq_lengths: lengths of sentences, batch_size, Long Tensor
        attention:
            score = v tanh(Wh+b)
            att = softmax(score)
        '''
        batch_size, max_len, _ = word_rep.size()
        # batch * max_len * hidden_states
        #A matrix for each position
        hidden_vecs = torch.zeros(batch_size, max_len, self.hidden_dim).type_as(word_rep)
        for i in range(max_len):
            j = i%self.max_step
            hidden_vecs[:, i] = word_rep[:, i].matmul(self.cnn[j])

        return hidden_vecs, hidden_vecs[:, -1]

#####################################################################
#Referring to the code at https://github.com/allanj/pytorch_neural_crf
#####################################################################
class NNCRF(nn.Module):

    def __init__(self, label_size, vocab_size, embedding_dim, hidden_dim, model_type='lstm'):
        super(NNCRF, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embedding_dim)

        ############################Modified########################################
        self.hidden2tag = nn.Linear(hidden_dim, label_size)
        if model_type == 'lstm':
            print('biLSTM')
            self.encoder = LSTM(embedding_dim, hidden_dim//2, bidirectional=True)
        elif model_type == 'gru':
            print('biGRU')
            self.encoder = GRU(embedding_dim, hidden_dim//2, bidirectional=True)
        elif model_type == 'elman':
            print('biElman')
            self.encoder = RNN(embedding_dim, hidden_dim//2, bidirectional=True)
        elif model_type == 'va_ew':
            self.encoder = VA_EW(embedding_dim, hidden_dim//2, bidirectional=True)
        elif model_type == 'mvmame':
            self.encoder = MVMA_ME(embedding_dim, hidden_dim//2, bidirectional=True)
        elif model_type == 'matmul':
            self.encoder = MatMultiModel(int(np.sqrt(embedding_dim)), hidden_dim, bidirectional=True)
            self.hidden2tag = nn.Linear(hidden_dim*2, label_size)
        else:
            self.encoder = VAWEncoder(embedding_dim, hidden_dim)
        
            
        #self.encoder = LSTM(embedding_dim, hidden_dim//2, bidirectional=True)
        
        #######################################################################
        self.inferencer = LinearCRF(label_size=label_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep = self.embedder(words)
        word_rep = self.dropout(word_rep)
        ###################Modification##########################
        #ltm_scores = self.encoder(word_rep, word_seq_lens.cpu())
        lstm_out, _ = self.encoder(word_rep, word_seq_lens)
        lstm_scores = self.hidden2tag(lstm_out)
        ##########################################################
        
        batch_size = words.size(0)
        sent_len = words.size(1)

        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len)
        maskTemp = maskTemp.to(curr_dev)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens.to(curr_dev), labels, mask.to(curr_dev))
        return unlabed_score - labeled_score

    def decode(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        word_rep = self.embedder(words)#, word_seq_lens)
        lstm_out, _ = self.encoder(word_rep, word_seq_lens)
        features = self.hidden2tag(lstm_out)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens.to(curr_dev))
        return bestScores, decodeIdx
    
    def load_pretrained_emb(self, path, is_training=False):
        '''
        Load pre-trained word embeddings from the path
        Arg:
            path: the binary file of local Glove embeddings
        '''
        with open(path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(path, vectors.shape))
            assert vectors.shape == self.embedder.weight.size()
            self.embedder.weight.data.copy_(torch.from_numpy(vectors))
            self.embedder.weight.requires_grad = is_training
            print('embeddings loaded')

class LinearCRF(nn.Module):


    def __init__(self, label_size:int):
        super(LinearCRF, self).__init__()

        self.label_size = label_size


        self.start_idx = label_size - 1#self.label2idx[START_TAG]
        self.end_idx = label_size - 2#self.label2idx[STOP_TAG]
        self.pad_idx = label_size - 3#self.label2idx[PAD]

        # initialize the following transition (anything never cannot -> start. end never  cannot-> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size)
#         init_transition = torch.zeros(self.label_size, self.label_size)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0


        self.transition = nn.Parameter(init_transition)



    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores=  self.calculate_all_scores(lstm_scores= lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabed_score, labeled_score

    def get_marginal_score(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        marginal = self.forward_backward(lstm_scores=lstm_scores, word_seq_lens=word_seq_lens)
        return marginal

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            # alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
            alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, dim=1)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        ## final score for the unlabeled network in this batch, with size: 1
        return torch.sum(last_alpha)

    def backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Backward algorithm. A benchmark implementation which is ready to use.
        :param lstm_scores: shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Backward variable
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        ## The code below, reverse the score from [0 -> length]  to [length -> 0].  (NOTE: we need to avoid reversing the padding)
        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        ## backward operation
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ## Following code is used to check the backward beta implementation
        last_beta = torch.gather(beta, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_beta += self.transition.transpose(0, 1)[:, self.start_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_beta = log_sum_exp_pytorch(last_beta.view(batch_size, self.label_size, 1)).view(batch_size)

        # This part if optionally, if you only use `last_beta`.
        # Otherwise, you need this to reverse back if you also need to use beta
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        return torch.sum(last_beta)

    def forward_backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Note: This function is not used unless you want to compute the marginal probability
        Forward-backward algorithm to compute the marginal probability (in log space)
        Basically, we follow the `backward` algorithm to obtain the backward scores.
        :param lstm_scores:   shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Marginal score. If you want probability, you need to use `torch.exp` to convert it into probability
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        dev_num = lstm_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        alpha = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)
        beta = torch.zeros(batch_size, seq_len, self.label_size, device=curr_dev)

        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        perm_idx = torch.zeros(batch_size, seq_len, device=curr_dev)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        alpha[:, 0, :] = scores[:, 0, self.start_idx, :]  ## the first position of all labels = (the transition from start - > all labels) + current emission.
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = alpha[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + scores[ :, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view( batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size, 1, 1).expand(batch_size, seq_len, self.label_size)

        ## Because we need to use the beta variable later, we need to reverse back
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        # `alpha + beta - last_alpha` is the standard way to obtain the marginal
        # However, we have two emission scores overlap at each position, thus, we need to subtract one emission score
        return alpha + beta - last_alpha - lstm_scores

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        dev_num = all_scores.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size], device=curr_dev)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64, device=curr_dev)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64, device=curr_dev)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64, device=curr_dev)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(curr_dev)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx

# if __name__ == '__main__':
#     import random
#     seed = 42
#     import numpy as np
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     import time
#     # test fast crf
#     labels = ['a', PAD, START_TAG, STOP_TAG]
#     label2idx = {'a': 0, PAD:1, START_TAG: 2, STOP_TAG: 3 }
#     model = LinearCRF(label_size=len(labels), label2idx=label2idx,
#                           idx2labels=labels)
#     seq_len = 80
#     batch_size = 5
#     all_lengths = torch.randint(1, seq_len, (batch_size,))
#     print(all_lengths)
#     # all_lengths = torch.LongTensor([7, 14])
#     all_scores = torch.randn(batch_size, max(all_lengths), len(labels), len(labels))
#     word_seq_lens = torch.LongTensor(all_lengths)
#     start = time.time()
#     output = model.forward_unlabeled(all_scores=all_scores, word_seq_lens=word_seq_lens)
#     end = time.time()
#     print(f"running time: {(end - start) * 1000}")
#     print(output)

    ##
#     print("##testing decoding process.")
#     with torch.no_grad():
#         scores, indices = model.viterbi_decode(all_scores=all_scores, word_seq_lens=word_seq_lens)
#     print(scores.squeeze(-1))
#     for idx in range(batch_size):
#         indices[idx][:word_seq_lens[idx]] = indices[idx][:word_seq_lens[idx]].flip([0])
#     for i, seq_len in enumerate(all_lengths):
#         print(indices[i, :seq_len])