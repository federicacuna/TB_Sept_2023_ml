#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq

from scipy.spatial.distance import euclidean
import itertools
import ast
import re
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import math


def tollerancex(row,sigma):
    diff = abs(row['x_hit'] - row['xMC'])
    threshold = sigma*row['dx_hit']
    if diff < threshold:
        return 1
    else:
        return 0
    
def tollerancey(row,sigma):
    diff = abs(row['y_hit'] - row['yMC'])
    threshold = sigma*row['dy_hit']
    if diff < threshold:
        return 1
    else:
        return 0
    
#ang x
def angx(row):
    diffx = row['x_hit_s'] - row['x_hit_t']
    diffz = row['zx_hit_s'] - row['zx_hit_t']
    angle = math.degrees(math.atan2(diffz, diffx))
    if angle < 0:
        angle += 360
    return angle

def angxMC(row):
    diffx = row['xMC_s'] - row['xMC_t']
    diffz = row['zx_hit_s'] - row['zx_hit_t']
    angle = math.degrees(math.atan2(diffz, diffx))
    if angle < 0:
        angle += 360
    return angle

def de_angx(row):
    diffx = row['x_hit_s'] - row['x_hit_t']
    diffz = row['zx_hit_s'] - row['zx_hit_t']
    dx1 =  row['dx_hit_s']**2
    dx2 =  row['dx_hit_t']**2
    den=diffx**2+diffz**2
    de_ang=(diffz/den)*math.sqrt(dx1+dx2)
  
    return math.degrees(de_ang)

#angy
def angy(row):
    diffy = row['y_hit_s'] - row['y_hit_t']
    diffz = row['zy_hit_s'] - row['zy_hit_t']
    angle = math.degrees(math.atan2(diffz, diffy))
    if angle < 0:
        angle += 360
    return angle

def angyMC(row):
    diffy = row['yMC_s'] - row['yMC_t']
    diffz = row['zy_hit_s'] - row['zy_hit_t']
    angle = math.degrees(math.atan2(diffz, diffy))
    if angle < 0:
        angle += 360
    return angle

def de_angy(row):
    diffy= row['y_hit_s'] - row['y_hit_t']
    diffz = row['zy_hit_s'] - row['zy_hit_t']
    dy1 =  row['dy_hit_s']**2
    dy2 =  row['dy_hit_t']**2
    den=diffy**2+diffz**2
    de_ang=(diffz/den)*math.sqrt(dy1+dy2) 
    return math.degrees(de_ang)

def angx_tollerance(row,sigma):
    if (row['angx_row']-row['angx_MC'])<sigma*row['deangx']:
        return 1
    else:
        return 0
    
def angy_tollerance(row,sigma):
    if (row['angy_row']-row['angy_MC'])<sigma*row['deangy']:
        return 1
    else:
        return 0    
    
    
def normalization(old_min,old_max,new_min,new_max,df,col_name):
    df[col_name] = np.interp(df[col_name], (old_min, old_max), (new_min, new_max))

old_min_xhit=-17.0
old_max_xhit=17.0
old_min_yhit=-17.0
old_max_yhit=17.0
old_min_PE=3
old_max_PE=30000
old_min_dx=0.1
old_max_dx=5.0
old_min_dy=0.1
old_max_dy=5.0
old_min_z=-29.0
old_max_z=1800.0
old_max_ang=360
old_min_ang=0
old_max_dxang=-6

    
if __name__ == "__main__":

    #InputFile
    data_folder=sys.argv[1]
    try:
        finx = sys.argv[2] #put parquet file xview 
        finy = sys.argv[3] #put parquet file yview 
    except:
        print("ERROR MESSAGE: \n =====> insert the data file name")
        sys.exit(1)
        
    data_trackingx=data_folder+finx+'.parquet'
    data_trackingy=data_folder+finy+'.parquet'
    print('data_x ',data_trackingx)
    print('data_x ',data_trackingy)
    save_data=int(sys.argv[4])
    primary_hit=int(sys.argv[5])
    print('primary_hit ',primary_hit)
    norm=0
    if(len(sys.argv)>6):
        norm=int(sys.argv[6])
        if(norm==1):
            print('you will normalize the features')
    
    
    #read data and create a pandas dataframe for nodes and edges
    dfx=pd.read_parquet(data_trackingx)
    dfx=dfx.drop(columns='view',axis=0)
    #df=df.sort_values(by=['Ev', 'ly'], ascending=[True, False])
    dfx = dfx.reset_index(drop=True)
    dfx=dfx.dropna(axis=0)
    #df=df[df.Ev==0]
    n_eventx = dfx['Ev'].unique()
    print('num of events x: ',n_eventx)
    #node_df
    dfx = dfx.groupby('Ev').filter(lambda x: x['ly'].nunique() > 3)###delete events where just a ly has bin hit
    
    dfy=pd.read_parquet(data_trackingy)
    dfy=dfy.drop(columns='view',axis=0)
    #df=df.sort_values(by=['Ev', 'ly'], ascending=[True, False])
    dfy = dfy.reset_index(drop=True)
    dfy=dfy.dropna(axis=0)
    #df=df[df.Ev==0]
    n_eventy = dfy['Ev'].unique()
    print('num of eventsy: ',n_eventy)
    #node_df
    dfy = dfy.groupby('Ev').filter(lambda x: x['ly'].nunique() > 3)###delete events where just a ly has bin hit


    node_dfx=dfx.apply(lambda x: x.explode() if x.name in ['x_hit','dx_hit','PE','PID'] else x)
    node_dfy=dfy.apply(lambda x: x.explode() if x.name in ['y_hit','dy_hit','PE','PID'] else x)

    events=node_dfx.Ev.unique()
    common_events = node_dfx[node_dfx['Ev'].isin(node_dfy['Ev'])]['Ev'].unique().tolist()
    node_dfx = node_dfx[node_dfx['Ev'].isin(common_events)]
    node_dfy = node_dfy[node_dfy['Ev'].isin(common_events)]


    #define the hit_class and add element to node_df dataset.
    node_dfx['x_hit'] = node_dfx['x_hit'].astype(float)
    node_dfx['dx_hit'] = node_dfx['dx_hit'].astype(float)
    #node_df['hit_class'] = node_df.apply(tollerancex, axis=1,args=(sigma,))
    node_dfx['hit_class'] = node_dfx['PID'].apply(lambda x: 1 if x == primary_hit else 0)
    node_dfx['idx_node'] = node_dfx.groupby('Ev').cumcount()
    node_dfx=node_dfx.dropna()
    #definition of max and min of the dataset
    min_xhit=node_dfx['x_hit'].min()
    min_zxhit=node_dfx['zx_hit'].min()
    min_PE=node_dfx['PE'].min()
    min_dx_hit=node_dfx['dx_hit'].min()
    max_xhit=node_dfx['x_hit'].max()
    max_zxhit=node_dfx['zx_hit'].max()
    max_PE=node_dfx['PE'].max()
    max_dx_hit=node_dfx['dx_hit'].max()

    node_dfy['y_hit'] = node_dfy['y_hit'].astype(float)
    node_dfy['dy_hit'] = node_dfy['dy_hit'].astype(float)
    #node_df['hit_class'] = node_df.apply(tollerancey, axis=1,args=(sigma,))
    node_dfy['hit_class'] = node_dfy['PID'].apply(lambda x: 1 if x == primary_hit else 0)
    node_dfy['idx_node'] = node_dfy.groupby('Ev').cumcount()
    node_dfy=node_dfy.dropna()
    #definition of max and min of the dataset
    min_yhit=node_dfy['y_hit'].min()
    min_zyhit=node_dfy['zy_hit'].min()
    min_PE=node_dfy['PE'].min()
    min_dy_hit=node_dfy['dy_hit'].min()
    max_yhit=node_dfy['y_hit'].max()
    max_zyhit=node_dfy['zy_hit'].max()
    max_PE=node_dfy['PE'].max()
    max_dy_hit=node_dfy['dy_hit'].max()

              
    #edge_df 


    edge_dfsx=[]
    for event, group in node_dfx.groupby('Ev'):
        # Itera attraverso i layer
        
        for layer in range(0,node_dfx['ly'].max()+1):
            
            # Seleziona i nodi corrispondenti al layer corrente e al layer successivo
            nodi_layer_corrente = group[group['ly'] == layer]
            nodi_layer_successivo = group[group['ly'] == layer +1]

            # Se i due layer hanno nodi
            if not nodi_layer_corrente.empty and not nodi_layer_successivo.empty:
                # Crea un DataFrame temporaneo degli edge
                edgesx = pd.merge(nodi_layer_corrente[['idx_node', 'x_hit', 'zx_hit','dx_hit','xMC','ly','PE','PID']], 
                                nodi_layer_successivo[['idx_node', 'x_hit', 'zx_hit','dx_hit','xMC','ly','PE','PID']], 
                                how='cross', suffixes=('_s', '_t'))
                
                # Aggiungi le colonne 'Ev' e rinomina le colonne 'idx_node' 
                edgesx['Ev'] = event
                edgesx = edgesx.rename(columns={'idx_node_source': 's_id', 'idx_node_target': 't_id'})
                #edges['ly']=layer
                # Aggiungi il DataFrame temporaneo alla lista
                edge_dfsx.append(edgesx)

    # Concatena tutti i DataFrame temporanei in un unico DataFrame degli edge
    df_edgesx = pd.concat(edge_dfsx, ignore_index=True)
    df_edgesx['angx_row'] = df_edgesx.apply(angx, axis=1)
    df_edgesx['angx_MC'] = df_edgesx.apply(angxMC, axis=1)
    # df_edgesx['deangx'] = df_edgesx.apply(de_angx, axis=1)
    # sigma=3
    #df_edges['edge_class'] = df_edges.apply(angx_tollerance, axis=1, args=(sigma,))
    min_angx=df_edgesx['angx_row'].min()
    max_angx=df_edgesx['angx_row'].max()
    # min_dangx=df_edgesx['deangx'].min()
    # max_dangx=df_edgesx['deangx'].max()
    if norm==1:
        '''
        node_df['x_hit']=node_df['x_hit']/100
        node_df['dx_hit']=node_df['dx_hit']/100
        node_df['zx_hit']=node_df['zx_hit']/82.46
        '''
        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
        #cols_node_df = ['x_hit', 'zx_hit','PE','dx_hit']
        #node_df[cols_node_df]=scaler.fit_transform(node_df[cols_node_df])
    
        normalization(old_min_xhit,old_max_xhit,0,1,node_dfx,'x_hit')
        normalization(old_min_z,old_max_z,0,1,node_dfx,'zx_hit')
        normalization(old_min_dx,old_max_dx,0,1,node_dfx,'dx_hit')
        normalization(old_min_PE,old_max_PE,0,1,node_dfx,'PE')
        normalization(0,3,0,1,node_dfx,'ly')
    
        # normalization(min_xhit,max_xhit,min_xhit,max_xhit,node_df,'x_hit')
        # normalization(min_zxhit,max_zxhit,min_zxhit,1789,node_df,'zx_hit')
        # normalization(min_dx_hit,max_dx_hit,min_dx_hit,max_dx_hit,node_df,'dx_hit')
        # normalization(min_PE,max_PE,0.1,max_PE,node_df,'PE')
            

    
    if norm==1:
        normalization(old_min_ang,old_max_ang,0,1,df_edgesx,'angx_row')
        # normalization(min_dangx,max_dangx,0,1,df_edges,'deangx')

    df_edgesx['edge_class'] = df_edgesx.apply(lambda row: 1 if row['PID_s'] == row['PID_t'] else 0, axis=1)
    df_edgesx=df_edgesx.dropna()

    
    edge_dfsy=[]
    for event, group in node_dfy.groupby('Ev'):
        # Itera attraverso i layer
        for layer in range(0,node_dfy['ly'].max()+1):
            
            # Seleziona i nodi corrispondenti al layer corrente e al layer successivo
            nodi_layer_corrente = group[group['ly'] == layer]
            nodi_layer_successivo = group[group['ly'] == layer + 1]

            # Se i due layer hanno nodi
            if not nodi_layer_corrente.empty and not nodi_layer_successivo.empty:
                # Crea un DataFrame temporaneo degli edge
                edgesy = pd.merge(nodi_layer_corrente[['idx_node', 'y_hit', 'zy_hit','dy_hit','yMC','ly','PE','PID']], 
                                nodi_layer_successivo[['idx_node', 'y_hit', 'zy_hit','dy_hit','yMC','ly','PE','PID']], 
                                how='cross', suffixes=('_s', '_t'))
                
                # Aggiungi le colonne 'Ev' e rinomina le colonne 'idx_node' 
                edgesy['Ev'] = event
                edgesy = edgesy.rename(columns={'idx_node_source': 's_id', 'idx_node_target': 't_id'})
                
                
                #edges['ly']=layer
                # Aggiungi il DataFrame temporaneo alla lista
                edge_dfsy.append(edgesy)

    # Concatena tutti i DataFrame temporanei in un unico DataFrame degli edge
    df_edgesy = pd.concat(edge_dfsy, ignore_index=True)
    df_edgesy['angy_row'] = df_edgesy.apply(angy, axis=1)
    df_edgesy['angy_MC'] = df_edgesy.apply(angyMC, axis=1)
    # df_edgesy['deangy'] = df_edgesy.apply(de_angy, axis=1)
    # sigma=3
    #df_edges['edge_class'] = df_edges.apply(angy_tollerance, axis=1,args=(sigma,))
    min_angy=df_edgesy['angy_row'].min()
    max_angy=df_edgesy['angy_row'].max()
    # min_dangy=df_edgesy['deangy'].min()
    # max_dangy=df_edgesy['deangy'].max()
    if norm==1:
        '''
        node_df['y_hit']=node_df['y_hit']/100
        node_df['dy_hit']=node_df['dy_hit']/100
        node_df['zy_hit']=node_df['zy_hit']/82.46
        '''
        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
        #cols_node_df = ['y_hit', 'zy_hit','PE','dy_hit']
        #node_df[cols_node_df]=scaler.fit_transform(node_df[cols_node_df])
        normalization(old_min_yhit,old_max_yhit,0,1,node_dfy,'y_hit')
        normalization(old_min_z,old_max_z,0,1,node_dfy,'zy_hit')
        normalization(old_min_dy,old_max_dy,0,1,node_dfy,'dy_hit')
        normalization(old_min_PE,old_max_PE,0,1,node_dfy,'PE')
        normalization(0,3,0,1,node_dfy,'ly')

    
    if norm==1:
        normalization(old_min_ang,old_max_ang,0,1,df_edgesy,'angy_row')
        # normalization(min_dangy,max_dangy,0,1,df_edgesy,'deangy')
    
    df_edgesy['edge_class'] = df_edgesy.apply(lambda row: 1 if row['PID_s'] == row['PID_t']==primary_hit else 0, axis=1)
    df_edgesy=df_edgesy.dropna()
       
    if(save_data==1 and norm==0):
        node_processedx='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/raw/'+finx+'node_df'+'.parquet'
        edge_processedx='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/raw/'+finx+'edge_df'+'.parquet'                                                    
        node_dfx_=pa.Table.from_pandas(node_dfx)
        edge_dfx_=pa.Table.from_pandas(df_edgesx)
        pq.write_table(node_dfx_,node_processedx)
        pq.write_table(edge_dfx_,edge_processedx)
        node_processedy='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/raw/'+finy+'node_df'+'.parquet'
        edge_processedy='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/raw/'+finy+'edge_df'+'.parquet'                                                    
        node_dfy_=pa.Table.from_pandas(node_dfy)
        edge_dfy_=pa.Table.from_pandas(df_edgesy)
        pq.write_table(node_dfy_,node_processedy)
        pq.write_table(edge_dfy_,edge_processedy)
        
    if(save_data==1 and norm==1):
        print('HEREEEE')
        print(f'/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/normalized/raw/{finx}node_df.parquet')
        node_processedx='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/normalized/raw/'+finx+'node_df'+'.parquet'
        edge_processedx='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/normalized/raw/'+finx+'edge_df'+'.parquet'                                                    
        node_dfx_=pa.Table.from_pandas(node_dfx)
        edge_dfx_=pa.Table.from_pandas(df_edgesx)
        pq.write_table(node_dfx_,node_processedx)
        pq.write_table(edge_dfx_,edge_processedx)
        
        node_processedy='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/normalized/raw/'+finy+'node_df'+'.parquet'
        edge_processedy='/lustrehome/federicacuna/TB_Sept_2023_ml/Data/preprocessed/normalized/raw/'+finy+'edge_df'+'.parquet'                                                   
        node_dfy_=pa.Table.from_pandas(node_dfy)
        edge_dfy_=pa.Table.from_pandas(df_edgesy)
        pq.write_table(node_dfy_,node_processedy)
        pq.write_table(edge_dfy_,edge_processedy)        