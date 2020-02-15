import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        
        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        #self.hidden = self.init_hidden()
        
     
    def forward(self, features, captions):
        # create embedded word vectors for each token in a batch of captions
        embeddings = self.word_embeddings(captions[:,:-1])  # batch_size,cap_length -> batch_size,cap_length-1,embed_size

         # -> batch_size, caption (sequence) length, embed_size
        embeddings = torch.cat([features.unsqueeze(1),embeddings.float()], dim=1)

        hiddens, _ = self.lstm(embeddings);   # print (lstm_out.shape) -> batch_size, caplength, hidden_size

        # get the scores for the most likely words
        outputs = self.linear(hiddens);     # print (outputs.shape) -> batch_size, caplength, vocab_size
        
        return outputs  #[:,:-1,:] # discard the last output of each sample in the batch.


       


    def sample(self, inputs, states=None, max_len=20):
        ''' accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)'''
        
        caption = []
        
        
        hidden = (torch.randn(self.num_layers, 1,self.hidden_dim).to(inputs.device),
                  torch.randn(self.num_layers, 1,self.hidden_dim).to(inputs.device))

        
        for i in range(max_len):
            
                lstm_out, hidden = self.lstm(inputs, hidden)
                outputs = self.linear(lstm_out)
                outputs = outputs.squeeze(1)
                wordid= outputs.argmax(dim=1)
                
                #print(f"i is {i} : ", outputs.size(), wordid.item())
                caption.append(wordid.item())
                
                inputs = self.word_embeddings(wordid.unsqueeze(0))
        return caption