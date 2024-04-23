from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import Sequential, GCNConv,GATConv,SAGEConv

    
class SageConv(torch.nn.Module):
    def __init__(self,num_feat,num_class):
        super().__init__()
        # Model layers
        HIDDEN_LAYER_SIZE=256
        self.conv1 = SAGEConv(num_feat, HIDDEN_LAYER_SIZE)
        self.conv2 = SAGEConv(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE-64)
        self.conv3 = SAGEConv(HIDDEN_LAYER_SIZE-64, num_class)
        #self.lin_1 = Linear(HIDDEN_LAYER_SIZE, NUM_CLASSES*4)
        #self.lin_2 = Linear(NUM_CLASSES*4, NUM_CLASSES)

    def forward(self, data):
        # The architecture itself
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.sigmoid(x)
###to be fixed       
# class SageConv2(torch.nn.Module):
#     def __init__(self,num_feat,num_class):
#         super().__init__()
#         # Model layers
#         hidd_size=448
#         self.conv1 = SAGEConv(num_feat, hidd_size)
#         self.conv2 = SAGEConv(hidd_size, hidd_size-64)
#         self.conv3 = SAGEConv(hidd_size-64, hidd_size-2*64)
#         self.conv4 = SAGEConv(hidd_size-2*64, hidd_size-3*64)
#         self.conv4 = SAGEConv(hidd_size-3*64, hidd_size-4*64)
#         self.conv4 = SAGEConv(hidd_size-4*64, hidd_size-5*64)
#         self.conv4 = SAGEConv(hidd_size-5*64, hidd_size-6*64)
#         self.conv4 = SAGEConv(hidd_size-5*64, hidd_size-6*64)
#         self.conv5 = SAGEConv(hidd_size-32, num_class)

#         #self.lin_1 = Linear(HIDDEN_LAYER_SIZE, NUM_CLASSES*4)
#         #self.lin_2 = Linear(NUM_CLASSES*4, NUM_CLASSES)

#     def forward(self, data):
#         # The architecture itself
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.conv1(x, edge_index)
#         x = F.tanh(x)
#         x = self.conv2(x, edge_index)
#         x = F.tanh(x)
#         x = self.conv3(x, edge_index)
#         x = F.tanh(x)
#         x = self.conv4(x, edge_index)
#         x = F.tanh(x)
#         x = self.conv5(x, edge_index)


       
#         #x = self.lin_1(x)
#        # x = F.tanh(x)
#         #x = self.lin_2(x)
#         return F.sigmoid(x)
  

class GDPModel(torch.nn.Module):
    def __init__(self, num_features=5, hidden_size=128,num_class=1,heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.num_class=num_class
        self.heads=heads
        self.convs1=GATConv(self.num_features, self.hidden_size, edge_dim = 1,heads=self.heads)
        self.convs2=GATConv(self.hidden_size*self.heads, self.num_class, edge_dim = 1,heads=1)
        
        # self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim = 2),
        #               GATConv(self.hidden_size, self.hidden_size, edge_dim = 2)]
        # self.linear = torch.nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        
        x, edge_index, edge_attr, batch= data.x, data.edge_index, data.edge_attr, data.batch
        # print(edge_index.device)
        # print(x.device)
        # print(edge_attr.device)
        # print(batch.device)
        x = self.convs1(x, edge_index,edge_attr)
        x=F.tanh(x)
        x = self.convs2(x, edge_index,edge_attr)
        # for conv in self.convs[:-1]:
        #     x = conv(x, edge_index, edge_attr) # adding edge features here!
        #     x = F.relu(x)
        #     x = F.dropout(x, training=self.training)
        # x = self.convs[-1](x, edge_index, edge_attr) # edge features here as well
        # x = self.linear(x)
        
        
        return F.sigmoid(x) 


class GDPModel2(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_class, heads,num_ly_val):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.num_class = num_class
        heads = heads
        num_ly_val = num_ly_val
        
        # Definisci una lista per memorizzare i tuoi strati GAT
        self.convs = torch.nn.ModuleList()
        
        # Aggiungi il primo strato GAT
        self.convs.append(GATConv(self.num_features, self.hidden_size, heads,edge_dim=1 ))
        
        # Aggiungi i layer intermedi GAT
        for _ in range(num_ly_val - 2):
            self.convs.append(GATConv(self.hidden_size * heads, self.hidden_size,  heads, edge_dim=1))
        
        # Aggiungi l'ultimo strato GAT
        self.convs.append(GATConv(self.hidden_size *heads, self.num_class, edge_dim=1, heads=1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Passaggio attraverso tutti gli strati GAT
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.tanh(x)
        
        # Applica la funzione sigmoide sull'output finale
        x = F.sigmoid(x)
        
        return x

        
class SageConv2(torch.nn.Module):
    def __init__(self, hidden_size, num_ly, num_feat, num_class,num_to_reduce):
        super().__init__()
        if(hidden_size - (num_ly - 1) * num_to_reduce==0):
            print('not proper number of ly and hidden size')
            num_ly=int(hidden_size/num_to_reduce)
            print(f' the number of layer will be set to: {num_ly}')
            
            # return
        # Model layers
        self.num_ly = num_ly
        self.conv_layers = torch.nn.ModuleList()
        
        self.conv_layers.append(SAGEConv(num_feat, hidden_size))
        
        for i in range(1, num_ly):
            current_hidden_size = hidden_size - i * num_to_reduce
            prev_hidden_size = hidden_size - (i - 1) * num_to_reduce
            self.conv_layers.append(SAGEConv(prev_hidden_size, current_hidden_size))

        self.conv_layers.append(SAGEConv(hidden_size - (num_ly - 1) * num_to_reduce, num_class))

    def forward(self, data):
    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = F.tanh(x)
            # x = F.relu(x)
        
        x = self.conv_layers[-1](x, edge_index)
        
        return F.sigmoid(x)

    
    
class GCN(torch.nn.Module):
    def __init__(self,num_feat,num_class):
        super().__init__()
        # Model layers
        hidd_size=256
        self.conv1 = GCNConv(num_feat, hidd_size)
        self.conv2 = GCNConv(hidd_size, hidd_size-64)
        self.conv3 = GCNConv(hidd_size-64, num_class)
        #self.lin_1 = Linear(HIDDEN_LAYER_SIZE, NUM_CLASSES*4)
        #self.lin_2 = Linear(NUM_CLASSES*4, NUM_CLASSES)

    def forward(self, data):
        # The architecture itself
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        x = self.conv3(x, edge_index)
       
        #x = self.lin_1(x)
       # x = F.tanh(x)
        #x = self.lin_2(x)
        return F.sigmoid(x)

class GCN_2(torch.nn.Module):
    def __init__(self, hidden_size, num_ly, num_feat, num_class,num_to_reduce):
        super().__init__()
        if(hidden_size - (num_ly - 1) * num_to_reduce==0):
            print('not proper number of ly and hidden size')
            num_ly=int(hidden_size/num_to_reduce)
            print(f' the number of layer will be set to: {num_ly}')
            
            # return
        # Model layers
        self.num_ly = num_ly
        self.conv_layers = torch.nn.ModuleList()
        
        self.conv_layers.append(GCNConv(num_feat, hidden_size))
        
        for i in range(1, num_ly):
            current_hidden_size = hidden_size - i * num_to_reduce
            prev_hidden_size = hidden_size - (i - 1) * num_to_reduce
            self.conv_layers.append(GCNConv(prev_hidden_size, current_hidden_size))

        self.conv_layers.append(GCNConv(hidden_size - (num_ly - 1) * num_to_reduce, num_class))

    def forward(self, data):
    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = F.tanh(x)
            # x = F.relu(x)
        
        x = self.conv_layers[-1](x, edge_index)
        
        return F.sigmoid(x)
    

class GAT(torch.nn.Module):
    def __init__(self,num_feat,num_class):
        super().__init__()
        # Model layers
        hidd_size=256
        heads=8
        self.conv1 = GATConv(num_feat, hidd_size,heads)
        #self.conv2 = GATConv(heads*(HIDDEN_LAYER_SIZE), HIDDEN_LAYER_SIZE,heads=8)
        #self.conv3 = GATConv(heads*(HIDDEN_LAYER_SIZE), NUM_CLASSES,heads=1)
        self.conv2 = GATConv(heads*hidd_size, num_class,heads=1)


    def forward(self, data):
        # The architecture itself
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
       # x = F.tanh(x)
       # x = self.conv3(x, edge_index)
     
        #x = self.lin_1(x)
       # x = F.tanh(x)
        #x = self.lin_2(x)
        return F.sigmoid(x)  