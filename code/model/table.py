import torch
from torch import nn
from .seq2mat import *
from .table_encoder.resnet import ResNet

class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        if config.seq2mat == 'tensor':
            self.seq2mat = TensorSeq2Mat(config)
        elif config.seq2mat == 'tensorcontext':
            self.seq2mat = TensorcontextSeq2Mat(config)
        elif config.seq2mat == 'context':
            self.seq2mat = ContextSeq2Mat(config)
        else:
            self.seq2mat = Seq2Mat(config)
        
        if config.table_encoder != 'none':
            self.layer = nn.ModuleList([ResNet(config) for _ in range(config.num_table_layers)])

    def forward(self, seq, mask):
        table = self.seq2mat(seq, seq)

        if self.config.table_encoder == 'none':
            return table

        for layer_module in self.layer:
            table = layer_module(table)

        return table
