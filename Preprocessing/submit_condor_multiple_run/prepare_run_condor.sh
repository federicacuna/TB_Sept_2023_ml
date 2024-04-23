#!/bin/bash
n1=$1
n2=$2



for ((i=n1; i<n2; i++)); do
    cat submit_Graph_cl_x_inCHANGEME.csi | sed "s/CHANGEME/$i/g" > submit_Graph_cl_x_in_$i.csi
    cat submit_Graph_cl_y_inCHANGEME.csi | sed "s/CHANGEME/$i/g" > submit_Graph_cl_y_in_$i.csi
    
done



for ((i=n1;i<n2;++i));do

    echo "submit_Graph_cl_x_in_$i"
    echo "submit_Graph_cl_y_in_$i"
    condor_submit "submit_Graph_cl_x_in_$i.csi" -name ettore ;
    condor_submit "submit_Graph_cl_y_in_$i.csi" -name ettore ;
done
