from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import torch
import numpy as np


class FastBERT():
    def __init__(self):
        databunch = BertDataBunch('train',
                                  'train',
                                  tokenizer='distilbert-base-uncased',
                                  train_file='train.csv',
                                  val_file='val.csv',
                                  label_file='labels.csv',
                                  text_col='text',
                                  label_col='label',
                                  batch_size_per_gpu=8,
                                  max_seq_length=512,
                                  multi_gpu=False,
                                  multi_label=False,
                                  model_type='distilbert')

        device_cuda = torch.device("cuda")
        metrics = [{'name': 'accuracy', 'function': accuracy}]
        logger = logging.getLogger()

        self.learner = BertLearner.from_pretrained_model(databunch,
                                                         pretrained_path='distilbert-base-uncased',
                                                         metrics=metrics,
                                                         device=device_cuda,
                                                         output_dir='models',
                                                         warmup_steps=100,
                                                         logger=logger,
                                                         multi_gpu=False,
                                                         is_fp16=False,  # install apex to use fp16 training
                                                         multi_label=False,
                                                         logging_steps=0)

    def train(self):
        self.learner.fit(epochs=5,
                         lr=6e-5,
                         validate=True,
                         schedule_type="warmup_cosine",
                         optimizer_type="lamb")

        self.learner.save_model()
