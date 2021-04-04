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
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first = True)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)    
        
    def forward(self, features, captions):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        batch_size = features.size(0)
        
        embeddings = self.embed(captions[:, :-1])
        
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        
        outputs, (self.hidden_state, self.cell_state) = self.lstm(embeddings, (self.hidden_state, self.cell_state))
        outputs = self.fc(outputs)
        
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        
        for i in range(max_len):
            
            output, states = self.lstm(inputs, states)
            
            output = self.fc(output)
            
            output = output.squeeze(1)
            _, max_index = torch.max(output, dim=1) 
            
            outputs.append(max_index.cpu().numpy()[0].item()) 
            
            if(max_index.cpu().numpy()[0].item() == 1):
                break
      
            inputs = self.embed(max_index) 
            inputs = inputs.unsqueeze(1)
            
        return outputs    