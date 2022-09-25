# Import libraries
import pandas as pd
import numpy as np
import re
import transformers
from transformers import BertModel,BertTokenizer,AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
import json
import torch
from torch import nn ,cuda
from torch.utils.data import DataLoader,Dataset,RandomSampler, SequentialSampler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bs4 import BeautifulSoup
import gc
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

from collections import Counter
import operator
from operator import itemgetter

pd.set_option('display.max_columns', 500)

df_desc = pd.read_csv('C:/Users/daryl/Desktop/PLP Proj/steam_with_description.csv', encoding='ISO-8859-1')
df_desc.head()
df_desc['steamspy_tags']= df_desc['steamspy_tags'].str.split(";")
exploded = df_desc.explode('steamspy_tags')
exploded['steamspy_tags'] = exploded['steamspy_tags'].str.replace(r'[^A-Za-z,]', '', regex=True).replace('', np.nan, regex=False)

# count of unique tags
print(len(exploded['steamspy_tags'].unique()))
print(exploded['steamspy_tags'].value_counts()[0:20])

exploded = exploded[exploded['steamspy_tags'] != "D"]
exploded = exploded[exploded['steamspy_tags'] != "EarlyAccess"]
exploded = exploded[exploded['steamspy_tags'] != "FreetoPlay"]

top20 = dict(Counter(exploded['steamspy_tags']).most_common(20))
print(top20)


names = list(top20.keys())
values = list(top20.values())

plt.barh(range(len(top20)), values, tick_label=names)
ax = plt.gca()
ax.invert_yaxis()
plt.title("Top 20 label occurences")
plt.show()

top_20_tags = [x for x in names if x == x]

def filterval(ls, val):
    if val in ls:
        ls.remove(val)
    return ls
filterval(top_20_tags, "D")
print(top_20_tags)

tags = exploded.groupby(['name', 'detailed_description']).apply(lambda x:x['steamspy_tags'].values).reset_index(name='final_tags')
tags['final_tags'] = tags['final_tags']
print(tags)

def pre_process(text):
    text = BeautifulSoup(text).get_text()

    # fetch alphabetic characters
    text = re.sub("[^a-zA-Z]", " ", text)

    # convert text to lower case
    text = text.lower()

    # split text into tokens to remove whitespaces
    tokens = text.split()

    return " ".join(tokens)

tags['clean_desc'] = tags['detailed_description'].apply(pre_process)

x=[] # To store the filtered clean_body values
y=[] # to store the corresponding tags
z=[] # store names


for i in range(len(tags['final_tags'])):
    temp=[]
    for tag in tags['final_tags'][i]:
        if tag in top_20_tags:
            temp.append(tag)

    if(len(temp)>0):
        z.append(tags['name'][i])
        x.append(tags['clean_desc'][i])
        y.append(temp)

final_df = pd.DataFrame(
    {'name': z,
     'desc': x,
     'tags': y
    })

print(final_df)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

yt = mlb.fit_transform(y)
yt.shape

# Getting a sense of how the tags data looks like
print(yt[0])
print(mlb.inverse_transform(yt[0].reshape(1,-1)))
print(mlb.classes_)

#dealing with class imbalance
class_counts = np.sum(yt,axis=0)
print(class_counts)
pos_weights = [(class_counts.sum() - x)/ (x+1e-5) for i, x in enumerate(class_counts)]
print(pos_weights)
pos_weights = torch.as_tensor(pos_weights, dtype=torch.float)

# compute no. of words in each desc
desc = x
word_cnt = [len(quest.split()) for quest in desc]
# Plot the distribution
plt.figure(figsize=[12,8])
plt.hist(word_cnt, bins = 40)
plt.xlabel('Word Count/Description')
plt.ylabel('# of Occurences')
plt.title("Frequency of Word Counts/Description")
plt.show()

from sklearn.model_selection import train_test_split
# First Split for Train and Test
x_train,x_test,y_train,y_test = train_test_split(desc, yt, test_size=0.1, random_state=RANDOM_SEED,shuffle=True)
# Next split Train in to training and validation
x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_SEED,shuffle=True)

len(x_tr) ,len(x_val), len(x_test)

class QTagDataset(Dataset):
    def __init__(self, quest, tags, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = quest
        self.labels = tags
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item_idx):
        text = self.text[item_idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,  # Add [CLS] [SEP]
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,  # Differentiates padded vs normal token
            truncation=True,  # Truncate data beyond max length
            return_tensors='pt'  # PyTorch Tensor format
        )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        # token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'label': torch.tensor(self.labels[item_idx], dtype=torch.float)

        }


class QTagDataModule(pl.LightningDataModule):

    def __init__(self, x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer, batch_size=32, max_token_len=300):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.train_dataset = QTagDataset(quest=self.tr_text, tags=self.tr_label, tokenizer=self.tokenizer,
                                         max_len=self.max_token_len)
        self.val_dataset = QTagDataset(quest=self.val_text, tags=self.val_label, tokenizer=self.tokenizer,
                                       max_len=self.max_token_len)
        self.test_dataset = QTagDataset(quest=self.test_text, tags=self.test_label, tokenizer=self.tokenizer,
                                        max_len=self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=16)

# Initialize the Bert tokenizer
BERT_MODEL_NAME = 'bert-base-cased' # we will use the BERT base model(the smaller one)
Bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

max_word_cnt = 500
desc_cnt = 0

# For every sentence...
for des in desc:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = Bert_tokenizer.encode(des, add_special_tokens=True)

    # Update the maximum sentence length.
    if len(input_ids) > max_word_cnt:
        desc_cnt +=1

print(f'# Description having word count > {max_word_cnt}: is  {desc_cnt}')

# Initialize the parameters that will be use for training
N_EPOCHS = 20
BATCH_SIZE = 16
MAX_LEN = 500
LR = 7e-05

# Instantiate and set up the data_module
QTdata_module = QTagDataModule(x_tr,y_tr,x_val,y_val,x_test,y_test,Bert_tokenizer,BATCH_SIZE,MAX_LEN)
QTdata_module.setup()

# Initialize the Bert tokenizer
BERT_MODEL_NAME = 'bert-base-cased' # we will use the BERT base model(the smaller one)
Bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
class QTagClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=20, steps_per_epoch=None, n_epochs=35, lr=7e-5):
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)  # outputs = number of labels
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weights)

    def forward(self, input_ids, attn_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)

        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return [optimizer], [scheduler]

# Initialize the parameters that will be use for training
N_EPOCHS = 20
BATCH_SIZE = 16
MAX_LEN = 500
LR = 7e-05



# Instantiate the classifier model
steps_per_epoch = len(x_tr)//BATCH_SIZE
model = QTagClassifier(n_classes=20, steps_per_epoch=steps_per_epoch,n_epochs=N_EPOCHS,lr=LR)

#if call from checkpoint
#model_path = "" #if calling from checkpoint, manually input model from lightning+logs folder
#model = QTagClassifier.load_from_checkpoint(model_path)

#Initialize Pytorch Lightning callback for Model checkpointing

# saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='QTag-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

# Instantiate the Model Trainer

#if resuming from checkpoint
#trainer = pl.Trainer(max_epochs = N_EPOCHS , gpus = 1, callbacks=[checkpoint_callback],resume_from_checkpoint = model_path, progress_bar_refresh_rate = 30)

#If training from scratch
trainer = pl.Trainer(max_epochs = N_EPOCHS , gpus = 1, callbacks=[checkpoint_callback], progress_bar_refresh_rate = 30)

# Train the Classifier Model
trainer.fit(model, QTdata_module)



# Evaluate the model performance on the test dataset
trainer.test(model,datamodule=QTdata_module)

# Visualize the logs using tensorboard.
#run in terminal if not using colab/jupyter
#%load_ext tensorboard
#%tensorboard --logdir lightning_logs/


len(y_test), len(x_test)

# Size of Test set
print(f'Number of Games = {len(x_test)}')

from torch.utils.data import TensorDataset

# Tokenize all questions in x_test
input_ids = []
attention_masks = []

for quest in x_test:
    encoded_quest = Bert_tokenizer.encode_plus(
        quest,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    # Add the input_ids from encoded question to the list.
    input_ids.append(encoded_quest['input_ids'])
    # Add its attention mask
    attention_masks.append(encoded_quest['attention_mask'])

# Now convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y_test)

# Set the batch size.
TEST_BATCH_SIZE = 64

# Create the DataLoader.
pred_data = TensorDataset(input_ids, attention_masks, labels)
pred_sampler = SequentialSampler(pred_data)
pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=TEST_BATCH_SIZE)

flat_pred_outs = 0
flat_true_labels = 0

# Put model in evaluation mode
model = model.to(device)  # moving model to cuda
model.eval()


# Tracking variables
pred_outs, true_labels = [], []
# i=0
# Predict
for batch in pred_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_attn_mask, b_labels = batch

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        pred_out = model(b_input_ids, b_attn_mask)
        pred_out = torch.sigmoid(pred_out)
        # Move predicted output and labels to CPU
        pred_out = pred_out.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # i+=1
        # Store predictions and true labels
        # print(i)
        # print(outputs)
        # print(logits)
        # print(label_ids)
    pred_outs.append(pred_out)
    true_labels.append(label_ids)

pred_outs[0][0]

# Combine the results across all batches.
flat_pred_outs = np.concatenate(pred_outs, axis=0)

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

flat_pred_outs.shape, flat_true_labels.shape

# define candidate threshold values
threshold = np.arange(0.5, 0.99, 0.01)
threshold


# convert probabilities into 0 or 1 based on a threshold value
def classify(pred_prob, thresh):
    y_pred = []
    for tag_label_row in pred_prob:
        temp = []
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1)  # Infer tag value as 1 (present)
            else:
                temp.append(0)  # Infer tag value as 0 (absent)
        y_pred.append(temp)

    return y_pred


from sklearn import metrics

scores = []  # Store the list of f1 scores for prediction on each threshold

# convert labels to 1D array
y_true = flat_true_labels.ravel()

for thresh in threshold:
    # classes for each threshold
    pred_bin_label = classify(flat_pred_outs, thresh)

    # convert to 1D array
    y_pred = np.array(pred_bin_label).ravel()

    scores.append(metrics.f1_score(y_true, y_pred))

# find the optimal threshold
opt_thresh = threshold[scores.index(max(scores))]
print(f'Optimal Threshold Value = {opt_thresh}')

# predictions for optimal threshold
y_pred_labels = classify(flat_pred_outs, opt_thresh)
y_pred = np.array(y_pred_labels).ravel()  # Flatten

print(metrics.classification_report(y_true, y_pred))

y_pred = mlb.inverse_transform(np.array(y_pred_labels))
y_act = mlb.inverse_transform(flat_true_labels)

df = pd.DataFrame({'Body': x_test, 'Actual Tags': y_act, 'Predicted Tags': y_pred})

df.sample(10)
df.to_csv('test_result.csv')

QTmodel = QTagClassifier.load_from_checkpoint(model_path)
def predict(question):
    text_enc = Bert_tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length= MAX_LEN,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'
    )
    outputs = QTmodel(text_enc['input_ids'], text_enc['attention_mask'])
    pred_out = outputs[0].detach().numpy()
    #print(f'Outputs = {outputs}')
    #print(f'Type = {type(outputs)}')
    #print(f'Pred Outputs = {pred_out}')
    #print(f'Type = {type(pred_out)}')
    #preds = np.round(pred_out)
    preds = [(pred > opt_thresh) for pred in pred_out ]
    #pred_list = [ round(pred) for pred in pred_logits ]
    preds = np.asarray(preds)
    #print(f'Predictions = {preds}')
    #print(f'Type = {type(preds)}')
    #print(mlb.classes_)
    new_preds = preds.reshape(1,-1).astype(int)
    #print(new_preds)
    pred_tags = mlb.inverse_transform(new_preds)
    #print(mlb.inverse_transform(np.array(new_preds)))
    return pred_tags

#model inference
description = "Elden Ring is an action role-playing game played in a third-person perspective with gameplay focusing on combat and exploration; " \
              "it features elements similar to those found in other games developed by FromSoftware, such as the Souls series, Bloodborne, and " \
              "Sekiro: Shadows Die Twice. Director Hidetaka Miyazaki explained that players start with a linear opening but eventually progress to " \
              "freely explore the Lands Between, including its six main areas, as well as castles, fortresses, and catacombs scattered throughout the " \
              "open world map. These main areas are interconnected through a central hub that players can access later in the game's progression—similar to" \
              "Firelink Shrine from Dark Souls—and are explorable using the character's mount as the main mode of transport, " \
              "although a fast travel system is an available option. Throughout the game, players encounter non-player characters (NPCs) and " \
              "enemies alike, including the demigods who rule each main area and serve as the game's main bosses.["

tags = predict(description)
if not tags[0]:
    print('This Game can not be associated with any known tag - Please review to see if a new tag is required ')
else:
    print(f'Following Tags are associated : \n {tags}')
