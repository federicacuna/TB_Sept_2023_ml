import ROOT
import numpy as np
import itertools
import awkward as ak

def linear_fit(n_events,x_hit,z_hit,dx_hit):
    sum_zz_ev=[]
    sum_z_ev=[]
    sum_zx_ev=[]
    sum_x_ev=[]
    sum_n_ev=[]
    chi2_ev=[]
    for i in range(0,n_events):
        # print(i)
        sum_zz = 0
        sum_z = 0.
        sum_zx = 0.
        sum_x = 0.
        sum_n = 0.
        chi2=0.0    
        for j in range(0,len(x_hit[i])):
            # print(i)
            sum_zz += (z_hit[i][j]* z_hit[i][j]) / (dx_hit[i][j] * dx_hit[i][j])
            sum_z +=z_hit[i][j]/(dx_hit[i][j] * dx_hit[i][j])
            sum_zx +=(z_hit[i][j]*x_hit[i][j])/(dx_hit[i][j] * dx_hit[i][j])
            sum_x +=x_hit[i][j]/(dx_hit[i][j] * dx_hit[i][j])
            sum_n +=1.0/(dx_hit[i][j] * dx_hit[i][j])
            
                
            # if sum_n<=0.1:         
            #     print(z_hit[i][j],' ',dx_hit[i][j],' ',x_hit[i][j],' sum_n ',sum_n)

        sum_zz_ev.append(sum_zz)
        sum_z_ev.append(sum_z)
        sum_zx_ev.append(sum_zx)
        sum_x_ev.append(sum_x)
        sum_n_ev.append(sum_n)
    

    det_ev=[]
    ax_ev=[]
    bx_ev=[]
    for i in range(0,n_events):
        det = round(sum_zz_ev[i],3) * round(sum_n_ev[i],3) - round(sum_z_ev[i],3) * round(sum_z_ev[i],3)
        if det==0:
        #     print('warning, maybe the GNN did not select any good hit, this is an empty event')
            
        #     print('ev ',i,' sum_zx_ev ',sum_zx_ev[i],' sum_zz_ev ',sum_zz_ev[i],' sum_z_ev ',sum_z_ev[i] ,' sum_x_ev ',sum_x_ev[i],' sum_n_ev ',sum_n_ev[i])
        #     print('ev ',i,' sum_zz_ev ',round(sum_zz_ev[i],3),' sum_z_ev ',sum_z_ev[i] ,' sum_n_ev ',round(sum_n_ev[i],3))
            print('ev no good ',i)
            continue
        ax = (sum_zx_ev[i] * sum_n_ev[i] - sum_z_ev[i] * sum_x_ev[i]) / det
        bx = (sum_zz_ev[i] * sum_x_ev[i] - sum_zx_ev[i] * sum_z_ev[i]) / det
        det_ev.append(det)
        ax_ev.append(ax)
        bx_ev.append(bx)
    return ax_ev,bx_ev



def read_root_viewx(file_name):
    file = ROOT.TFile(file_name, "READ")
    tree = file.Get("Reco_tree")
    
    # Definisci la variabile per contenere il vettore
    ly_ev_vec = ROOT.std.vector("float")()
    x_ev_vec = ROOT.std.vector("float")()
    z_ev_vec = ROOT.std.vector("float")()
    dx_ev_vec = ROOT.std.vector("float")()
    pe_ev_vec = ROOT.std.vector("float")()
    no_good_ev_vec= ROOT.std.vector("int")()
    # Ottieni il branch ly_ev_vec
    tree.SetBranchAddress("ly_ev_vec", ROOT.AddressOf(ly_ev_vec))
    tree.SetBranchAddress("x_ev_vec", ROOT.AddressOf(x_ev_vec))
    tree.SetBranchAddress("z_ev_vec", ROOT.AddressOf(z_ev_vec))
    tree.SetBranchAddress("dx_ev_vec", ROOT.AddressOf(dx_ev_vec))
    tree.SetBranchAddress("pe_ev_vec", ROOT.AddressOf(pe_ev_vec))
    tree.SetBranchAddress("no_good_ev_vec", ROOT.AddressOf(no_good_ev_vec))
    
    
    # Leggi l'intero albero e ottieni il numero di eventi
    n_events = tree.GetEntries()
    n_events=5000
    print(n_events)
    
    # Itera su tutti gli eventi e stampa il vettore per ciascuno
    ly_hit=[]
    x_hit=[]
    z_hit=[]
    dx_hit=[]
    pe_hit=[]
    no_good_events=[]
    
    for i in range(0,n_events):
        tree.GetEntry(i)  # Leggi l'evento corrente  
        # if i%1000==0:
        #     print('processing ev ',i)
        # Stampa i valori nella lista
        no_good_events.append(list(no_good_ev_vec))
        
        if i not in list(no_good_ev_vec):   
            ly_hit.append([])
            x_hit.append([])
            z_hit.append([])
            dx_hit.append([])
            pe_hit.append([])            
            for ly,x,z,dx,pe in zip(ly_ev_vec,x_ev_vec,z_ev_vec,dx_ev_vec,pe_ev_vec):
                ly_hit[-1].append(ly)
                x_hit[-1].append(x)
                z_hit[-1].append(z)
                dx_hit[-1].append(dx)
                pe_hit[-1].append(pe)
        else:
            print('no good event ',i)
            # else:
            #     print( 'GNN did not find good hits in this event: ',i)
                # print('ev ',i,' ly  ',ly,' x ',x, ' z ',z,' dx ',dx)     
    # Chiudi il file ROOT
    
    file.Close()
    print(len(ly_hit))
    return ly_hit,x_hit,z_hit,dx_hit,pe_hit



def read_root_viewy(file_name):
    file = ROOT.TFile(file_name, "READ")
    tree = file.Get("Reco_tree")
    
    # Definisci la variabile per contenere il vettore
    ly_ev_vec = ROOT.std.vector("float")()
    y_ev_vec = ROOT.std.vector("float")()
    z_ev_vec = ROOT.std.vector("float")()
    dy_ev_vec = ROOT.std.vector("float")()
    pe_ev_vec = ROOT.std.vector("float")()
    no_good_ev_vec= ROOT.std.vector("int")()
    
    # Ottieni il branch ly_ev_vec
    tree.SetBranchAddress("ly_ev_vec", ROOT.AddressOf(ly_ev_vec))
    tree.SetBranchAddress("y_ev_vec", ROOT.AddressOf(y_ev_vec))
    tree.SetBranchAddress("z_ev_vec", ROOT.AddressOf(z_ev_vec))
    tree.SetBranchAddress("dy_ev_vec", ROOT.AddressOf(dy_ev_vec))
    tree.SetBranchAddress("pe_ev_vec", ROOT.AddressOf(pe_ev_vec))
    tree.SetBranchAddress("no_good_ev_vec", ROOT.AddressOf(no_good_ev_vec))

    
    # Leggi l'intero albero e ottieni il numero di eventi
    n_events = tree.GetEntries()
    n_events=5000
    print(n_events)
   
    # Itera su tutti gli eventi e stampa il vettore per ciascuno
    ly_hit=[]
    y_hit=[]
    z_hit=[]
    dy_hit=[]
    pe_hit=[]
    no_good_events=[]
    for i in range(0,n_events):
        tree.GetEntry(i)  # Leggi l'evento corrente  
        # Stampa i valori nella lista
        no_good_events.append(list(no_good_ev_vec))
        if i not in list(no_good_ev_vec):
           
            ly_hit.append([])
            y_hit.append([])
            z_hit.append([])
            dy_hit.append([])
            pe_hit.append([])            
            for ly,y,z,dy,pe in zip(ly_ev_vec,y_ev_vec,z_ev_vec,dy_ev_vec,pe_ev_vec):
                ly_hit[-1].append(ly)
                y_hit[-1].append(y)
                z_hit[-1].append(z)
                dy_hit[-1].append(dy)
                pe_hit[-1].append(pe)
        else:
            print('no good event y ',i)

        # else:
        #     print( 'GNN did not find good hits in this event: ',i)
              
    # Chiudi il file ROOT
    file.Close()
    print(len(ly_hit))
    return ly_hit,y_hit,z_hit,dy_hit,pe_hit



def lin_calc(xpos_x_comb,xpos_error_x_comb,zpos_x_comb):
    tmpden=xpos_error_x_comb*xpos_error_x_comb
    xsum_x_comb = ak.sum((xpos_x_comb)/(tmpden), axis=-1)
    zsum_x_comb = ak.sum((zpos_x_comb)/(tmpden), axis=-1)
    xzsum_x_comb = ak.sum((xpos_x_comb*zpos_x_comb)/(tmpden), axis=-1)
    zzsum_x_comb = ak.sum((zpos_x_comb*zpos_x_comb)/(tmpden), axis=-1)
    sum_n_x_comb = ak.sum(1./(tmpden), axis=-1)
    det_x_comb = zzsum_x_comb * sum_n_x_comb - zsum_x_comb * zsum_x_comb 
    a_x_comb = (xzsum_x_comb * sum_n_x_comb - xsum_x_comb * zsum_x_comb)/det_x_comb
    b_x_comb = (xsum_x_comb * zzsum_x_comb - xzsum_x_comb * zsum_x_comb)/det_x_comb
    chi2_x_comb = ak.sum((xpos_x_comb - a_x_comb * zpos_x_comb - b_x_comb)*(xpos_x_comb - a_x_comb * zpos_x_comb - b_x_comb)/(xpos_error_x_comb*xpos_error_x_comb), axis=-1)
    return a_x_comb,b_x_comb,chi2_x_comb


def evaluateRecoCosineDirector(Mx,My):
    a0 = pow(Mx,2.)
    a1 = pow(My,2.)
    crz = 1/(pow(a0+a1+1,0.5))
    crx = crz*Mx
    cry = crz*My
    return crx,cry,crz
