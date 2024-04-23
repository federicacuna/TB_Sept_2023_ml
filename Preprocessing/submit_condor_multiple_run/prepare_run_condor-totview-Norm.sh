#!/bin/bash
n1=$1
n2=$2



for ((i=n1; i<n2; i++)); do
    cat NORMsubmit_Graph_cl_inCHANGEME.csi | sed "s/CHANGEME/$i/g" > NORMsubmit_Graph_cl_in_$i.csi


done



for ((i=n1;i<n2;++i));do

    echo "NORMsubmit_Graph_cl_in_$i"

    condor_submit "NORMsubmit_Graph_cl_in_$i.csi" -name ettore ;

done
