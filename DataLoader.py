# A simple data loading class
# input type: numpy array
# output type: numpy array / tensor

import numpy as np

class DataLoader:
    def __init__(self, content, length=-1, input_max_len=-1, output_max_len=-1):
        
        # single label model
        if length == -1:
            self.seq_model = 0
            assert input_max_len == -1
            assert output_max_len == -1
            self.comb = content
        # sequence model
        else:
            self.seq_model = 1
            self.comb = np.concatenate((content,length),axis=1)
            self.input_max_len = input_max_len
            self.output_max_len = output_max_len
            
        self.dataLength = len(content)
        self.seqLength = len(content[0])
        self.pointer = 0
        
        print("[DataLoader]: initialized successfully")
        print("[DataLoader]: dataset size --", self.dataLength)
        
    def reset(self, batch_size = 0):
        self.pointer = 0
        self.batch_size = self.dataLength if batch_size == 0 else batch_size
        np.random.shuffle(self.comb)
        
        
    def getMiniBatch(self):
        if self.pointer + self.batch_size <= self.dataLength:
            lower, upper = self.pointer, self.pointer + self.batch_size
            self.pointer = upper
            inputs, lengths = self.comb[lower:upper, :self.seqLength], self.comb[lower:upper, self.seqLength:]
            
            if not self.seq_model:
                return inputs[:, :-1], inputs[:, -1:], lengths
            else:
                # sorting codes reference: https://blog.csdn.net/lssc4205/article/details/79474735
                inputs_tensor = torch.LongTensor(inputs)
                lengths_tensor = torch.LongTensor(lengths)

                if USE_CUDA:
                    inputs_tensor = inputs_tensor.cuda()
                    lengths_tensor = lengths_tensor.cuda()

                _, idx_sort = torch.sort(lengths_tensor[:,0], dim=0, descending=True)
                _, idx_unsort = torch.sort(idx_sort, dim=0)
                inputs_tensor = inputs_tensor.index_select(0, idx_sort)
                inputs_enc, inputs_dec = inputs_tensor[:,:self.input_max_len], inputs_tensor[:,self.input_max_len:]
                lengths_tensor = lengths_tensor[idx_sort]
                return inputs_enc, inputs_dec, lengths_tensor
        else: 
            return None