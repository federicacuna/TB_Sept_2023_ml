from torch_geometric.data import Dataset, Data
import os.path as osp
import torch
import pandas as pd
import numpy as np
import time
from dask.distributed import Client, LocalCluster
import dask.bag as db
import multiprocessing as mp
import dask.delayed    
#data_raw and data_prep are dctionary

class dataset_preparation():
    def __init__(self,data_raw=None,data_prep=None,root=None,n_workers=None):
        self.data_raw=data_raw
        self.data_prep=data_prep
        self.root=root
        self.n_workers=n_workers
    
    def get_index_data(self):
        idx_to_process=[]
        for indx in self.data_raw.keys():
            if indx not in self.data_prep.keys():
                idx_to_process.append(indx)
        # print('idx not processed ',self.data_raw.keys(),' idx_processed ',self.data_prep.keys() )
        return idx_to_process

    
    
    def process(self,idx):     
        node_file_path = self.data_raw[idx]['nodes']
        edge_file_path = self.data_raw[idx]['edges']
        data_n = pd.read_parquet(node_file_path)
        data_e = pd.read_parquet(edge_file_path)
        #print(data_n.head())
        ev = data_n.Ev.unique()
        data_list = []
        for i in ev:
            node_feat = data_n[data_n.Ev == i][["x_hit", "dx_hit", "ly","zx_hit","PE"]]
            xi = node_feat.to_numpy()
            xi = torch.tensor(xi, dtype=torch.float32)
            edge_feat=data_e[data_e.Ev==i][["angx_row","deangx"]]
            edge_feat=edge_feat.to_numpy()
            edge_feat=torch.tensor(edge_feat,dtype=torch.float32)
            labels = data_n[data_n.Ev == i]["hit_class"]
            yi = labels.to_numpy()
            yi = torch.tensor(yi, dtype=torch.int)

            edge_indexi = data_e[data_e.Ev == i][['idx_node_s', 'idx_node_t']]
            edge_indexi = np.array(edge_indexi).transpose()
            edge_indexi = torch.tensor(edge_indexi, dtype=torch.int64)

            data_i = Data(x=xi, edge_index=edge_indexi, edge_attr=edge_feat,y=yi)
            data_list.append(data_i)
        torch.save(data_list, osp.join(self.root+'/processed/', f'data_{idx}.pt'))

    def process_y(self,idx):     
        node_file_path = self.data_raw[idx]['nodes']
        edge_file_path = self.data_raw[idx]['edges']
        data_n = pd.read_parquet(node_file_path)
        data_e = pd.read_parquet(edge_file_path)
        #print(data_n.head())
        ev = data_n.Ev.unique()
        data_list = []
        for i in ev:
            node_feat = data_n[data_n.Ev == i][["y_hit", "dy_hit", "ly","zy_hit","PE"]]
            xi = node_feat.to_numpy()
            xi = torch.tensor(xi, dtype=torch.float32)
            edge_feat=data_e[data_e.Ev==i][["angy_row","deangy"]]
            edge_feat=edge_feat.to_numpy()
            edge_feat=torch.tensor(edge_feat,dtype=torch.float32)
            labels = data_n[data_n.Ev == i]["hit_class"]
            yi = labels.to_numpy()
            yi = torch.tensor(yi, dtype=torch.int)

            edge_indexi = data_e[data_e.Ev == i][['idx_node_s', 'idx_node_t']]
            edge_indexi = np.array(edge_indexi).transpose()
            edge_indexi = torch.tensor(edge_indexi, dtype=torch.int64)

            data_i = Data(x=xi, edge_index=edge_indexi, edge_attr=edge_feat,y=yi)
            data_list.append(data_i)
        torch.save(data_list, osp.join(self.root+'/processed/', f'data_{idx}.pt'))

    def process_on_cluster(self):
      
        idx_to_process=self.get_index_data()
        print("idx_to_process ",idx_to_process)
        if(len(idx_to_process)==0):
            print('already done')
            pass
        
        if  len(idx_to_process)==1:
            print('just single file, no need to process on cluster ')
            self.process(idx_to_process[0])
            print('standard process')
            
        if(len(idx_to_process)>1):
            print('use_cluster with ',self.n_workers, 'n_workers')
            with LocalCluster(n_workers=self.n_workers, 
            processes=True,
            threads_per_worker=1
        ) as cluster, Client(cluster) as client:
                b = db.from_sequence(idx_to_process).map(self.process).compute()


    def process_on_cluster_y(self):
      
        idx_to_process=self.get_index_data()
        print("idx_to_process ",idx_to_process)
        if(len(idx_to_process)==0):
            print('already done')
            pass
        
        if  len(idx_to_process)==1:
            print('just single file, no need to process on cluster ')
            self.process_y(idx_to_process[0])
            print('standard process')
            
        if(len(idx_to_process)>1):
            print('use_cluster with ',self.n_workers, 'n_workers')
            with LocalCluster(n_workers=self.n_workers, 
            processes=True,
            threads_per_worker=1
        ) as cluster, Client(cluster) as client:
                b = db.from_sequence(idx_to_process).map(self.process_y).compute()
        
    def get(self, idx,folder):
        data_list = torch.load(osp.join(self.root+'/processed/'+folder, f'data_{idx}.pt'))
        print('taking the ', f'data_{idx}.pt' )
        return data_list
    
    def get_more_file(self, idx_start,idx_stop,folder):
        data=[]
        for idx in range(idx_start,idx_stop):
            try:
                file_path = osp.join(self.root+'/processed/'+folder, f'data_{idx}.pt')
                data.extend(torch.load(file_path))
            except:
                print('file',idx, 'not found')
        print('taking a list of file from ', idx_start, ' to ' , idx_stop  )
        #print('len ',len(data))
        return data
    
   

    def load_file(self,file_path):
        try:
            return torch.load(file_path)
        except FileNotFoundError:
            print('file', file_path, 'not found')
            return []

    def get_more_files_dask(self, idx_start, idx_stop, folder):
        file_paths = [osp.join(self.root+'/processed/'+folder, f'data_{idx}.pt') for idx in range(idx_start, idx_stop)]
        # print(file_paths)
        with LocalCluster(n_workers=self.n_workers, 
            processes=True,
            threads_per_worker=1
        )as cluster, Client(cluster) as client:
            b = db.from_sequence(file_paths).map(self.load_file)
            data=b.compute()
        print('taking a list of file from ', idx_start, ' to ', idx_stop)
        return data
    
     