import logging

from collections import namedtuple

import numpy as np
import pandas as pd

# Global feature details
feature_names = ['ly', 'x_hit', 'zx_hit','PE','dx_hit']
# feature_scale = np.array([1000., np.pi, 1000.])

# Graph is a namedtuple of (X, Ri, Ro, y) for convenience
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])
# Sparse graph uses the indices for the Ri, Ro matrices
SparseGraph = namedtuple('SparseGraph',
        ['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y'])

def make_sparse_graph(X, Ri, Ro, y):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y)

def graph_from_sparse(sparse_graph, dtype=np.uint8):
    n_nodes = sparse_graph.X.shape[0]
    n_edges = sparse_graph.Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[sparse_graph.Ri_rows, sparse_graph.Ri_cols] = 1
    Ro[sparse_graph.Ro_rows, sparse_graph.Ro_cols] = 1
    return Graph(sparse_graph.X, Ri, Ro, sparse_graph.y)

def construct_graph(hits, segments):
    """Construct one graph (e.g. from one event)"""
    # Construct segments
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    evtid = hits.Ev.unique()
    # Prepare the tensors
    # X = (hits[feature_names].values / feature_scale).astype(np.float32)
    
    X = (hits[feature_names].values).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)

    seg_start = segments.idx_node_s.values
    seg_end = segments.idx_node_t.values
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = segments.PID_s.values
    pid2 = segments.PID_t.values
    y[:] = (pid1 == pid2)
    return make_sparse_graph(X, Ri, Ro, y)
    
    
def construct_graphs(hits, segments,
                     max_events=None):
 
    # Organize hits by event
    evt_hit_groups = hits.groupby('Ev')
    evt_seg_groups = segments.groupby('Ev')
    evtids = hits.Ev.unique()
    if max_events is not None:
        evtids = evtids[:max_events]
    
    # Loop over events and construct graphs
    graphs = []
    for evtid in evtids:
        # print(evtid)
        if (evtid not in hits['Ev'].values) or (evtid not in segments['Ev'].values):
            continue
        # Get the hits for this event
        evt_hits = evt_hit_groups.get_group(evtid)
        evt_segments=evt_seg_groups.get_group(evtid)
        graph = construct_graph(evt_hits,evt_segments)
        graphs.append(graph)

    # Return the results
    return graphs

def graph_from_files(list_file_node,list_file_edge,
                     max_events=None):
    
    graphs=[]
    for i,j in zip(list_file_node,list_file_edge):
        print(i)
        node_df=pd.read_parquet(i)
        edge_df=pd.read_parquet(j)
        # print(node_df)
        graphs.append(construct_graphs(node_df,edge_df))
    return graphs
        

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph._asdict())
    #np.savez(filename, X=graph.X, Ri=graph.Ri, Ro=graph.Ro, y=graph.y)

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def save_graph_combined(filename,graph):
    combined_data = {}
    
    # Aggiungi i dati di ciascun grafico al dizionario
    for i, graph in enumerate(graph):
        for key, value in graph._asdict().items():
            # Crea un nome univoco per ogni campo utilizzando l'indice del grafico
            new_key = f'graph_{i}_{key}'
            combined_data[new_key] = value
    
    # Salva i dati combinati in un unico file .npz
    np.savez(filename, **combined_data)

    
def load_graph(filename, graph_type=Graph):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return graph_type(**dict(f.items()))

def load_graphs(filenames, foldername, graph_type=Graph):
    return [load_graph(foldername+f, graph_type) for f in filenames]

def load_graph_combined(filename):
    """Read multiple graph NPZ files"""
    # Carica il file .npz
    with np.load(filename) as f:
        # Inizializza una lista per contenere i grafici
        graphs = []
        # Ottieni il numero di eventi nel file
        num_events = sum(1 for key in f.keys() if key.startswith('graph_'))
        
        # Itera attraverso gli eventi nel file e crea un grafico per ciascun evento
        for i in range(num_events):
            # Inizializza un dizionario per contenere i dati del grafico corrente
            graph_data = {}
            # Estrai i dati del grafico corrente
            for key in f.keys():
                if not key.startswith(f'graph_{i}_'):
                    continue
                graph_data[key.split('_', 2)[-1]] = f[key]
            
            # Verifica se tutti i campi necessari sono presenti
            if len(graph_data) != 4:
                continue
            
            # Estrai i dati del grafico corrente
            X = graph_data.get('X')
            Ri = graph_data.get('Ri')
            Ro = graph_data.get('Ro')
            y = graph_data.get('y')
            
            # Se uno dei campi manca, salta l'iterazione
            if X is None or Ri is None or Ro is None or y is None:
                continue
            
            # Crea un'istanza di Graph utilizzando i dati estratti
            graph = Graph(X=X, Ri=Ri, Ro=Ro, y=y)
            graphs.append(graph)
        
    return graphs