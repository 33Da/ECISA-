from torch.utils.data import DataLoader
import os
from torch.utils import data
import pandas as pd
import torch
import numpy as np
import re
from tqdm import tqdm
import pickle
from torch import nn
import torch.nn.functional as F
from torch import optim

max_len = 250
train_batch_size = 64
test_batch_size = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESISADataset(data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item["text"]
        label = item["label"]

        return text, label

# 读取数据
train_dataset = ESISADataset("data/SMP2019_ECISA/train.csv")
test_dataset = ESISADataset("data/SMP2019_ECISA/test.csv")
dev_dataset = ESISADataset("data/SMP2019_ECISA/dev.csv")

def tokenize(text):

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]

"""文本序列化"""
# Word2Sequence
class Word2Sequence:
    # 未出现过的词
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    # 填充的词
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.fited = False
        self.count = {}

    def to_index(self, word):
        """word -> index"""
        return self.dict.get(word, self.UNK)

    def to_word(self, index):
        """index -> word"""
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self, sentence):
        """count字典中存储每个单词出现的次数"""
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=None, max_count=None, max_feature=None):
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])
        # 给词典中每个词分配一个数字ID
        for word in self.count:
            self.dict[word] = len(self.dict)
        # 构建一个数字映射到单词的词典，方法反向转换，但程序中用不太到
        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
            构建词典
            只筛选出现次数在[min_count,max_count]之间的词
            词典最大的容纳的词为max_feature，按照出现次数降序排序，要是max_feature有规定，出现频率很低的词就被舍弃了
         """
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        # 截断文本
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r, dtype=np.int64)

    def inverse_transform(self, indices):
        """数字序列-->单词序列"""
        sentence = []
        for i in indices:
            word = self.to_word(i)
            sentence.append(word)
        return sentence

# 建立词表
def fit_save_word_sequence():
    word_to_sequence = Word2Sequence()
    global max_len
    # tqdm是显示进度条的
    for text,label in tqdm(train_dataset, ascii=True, desc="fitting"):

        max_len = max(max_len,len(text))
        word_to_sequence.fit(tokenize(text.strip()))
    word_to_sequence.build_vocab()
    # 对wordSequesnce进行保存
    pickle.dump(word_to_sequence, open("model/ws.pkl", "wb"))



# fit_save_word_sequence()


# 自定义的collate_fn方法
ws = pickle.load(open("./model/ws.pkl", "rb"))
def collate_fn(batch):
    batch = list(zip(*batch))
    labels = torch.tensor(batch[1], dtype=torch.int32)
    texts = batch[0]
    texts = torch.tensor([ws.transform(i, max_len) for i in texts])
    del batch
    return labels.long(), texts.long()

# # 测试数据集的功能
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
#
# for idx, (label, text) in enumerate(train_dataloader):
#     print("idx：", idx)
#     print("lable:", label)
#     print("text:", text)
#     break

# 获取数据的方法
def get_dataloader(train=True):
    if train:
        dataset = ESISADataset("data/SMP2019_ECISA/train.csv")
    else:
        dataset = ESISADataset("data/SMP2019_ECISA/test.csv")
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)




"""构建模型"""
class MY_BILSTM(nn.Module):
    def __init__(self):
        super(MY_BILSTM, self).__init__()
        self.hidden_size = 64
        self.embedding_dim = 200
        self.num_layer = 2
        self.bidirectional = True
        self.bi_num = 2 if self.bidirectional else 1
        self.dropout = 0.5
        # 以上部分为超参数，可以自行修改
        self.embedding = nn.Embedding(len(ws), self.embedding_dim, padding_idx=ws.PAD)  # 多少個詞  每個詞多少維  補充多少padding
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size,
                            self.num_layer, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * self.bi_num, 20)

        self.fc2 = nn.Linear(20, 3)


    def forward(self, x):
        x = self.embedding(x)  # 轉成每個詞用兩百個浮點數表示 0-1
        x = x.permute(1, 0, 2)  # 进行轴交换
        h_0, c_0 = self.init_hidden_state(x.size(1))
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # 只要最后一个lstm单元处理的结果，取前向LSTM和后向LSTM的结果进行简单拼接
        out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc2(out)


        return F.log_softmax(out, dim=-1)

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size).to(device)
        return h_0, c_0

imdb_model = MY_BILSTM()
# imdb_model.load_state_dict(torch.load("model/bilstm0.pkl"))
optimizer = optim.Adam(imdb_model.parameters())
criterion = nn.CrossEntropyLoss()



# 測試和訓練
def train(epoch):
    mode = True
    train_dataloader = get_dataloader(mode)
    for idx, (target, input) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = imdb_model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(input), len(train_dataloader.dataset),
                           100. * idx / len(train_dataloader), loss.item()))
            torch.save(imdb_model.state_dict(), f"model/bilstm{epoch}.pkl")
            torch.save(optimizer.state_dict(), f'model/optimizer_bilstm{epoch}.pkl')




def test():
    test_loss = 0
    correct = 0
    mode = False
    imdb_model.eval()
    test_dataloader = get_dataloader(mode)
    with torch.no_grad():
        for target, input in test_dataloader:
            output = imdb_model(input)
            test_loss += F.nll_loss(output, target, reduction="sum")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))

# 训练和测试
for i in range(10):

    train(i)
    print("训练第{}轮的测试结果------------------------------------".format(i + 1))
test()