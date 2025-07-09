# HIDTA

## Traffic Network
This experiment is conducted on multiple networks of increasing complexity (Sioux Falls, Chicago, and Anaheim). The city road network data is sourced from (https://github.com/bstabler/TransportationNetworks).  

The DTALITE is sourced from https://github.com/xzhou99/Dtalite_NeXTA_package


## Experiments

The experiments compared with the Baseline were conducted using DTAlite software, while the Baseline comes from the paper "Heterogeneous Graph Sequence Neural Networks for Dynamic Traffic Assignment". 
Ablation experiments and parameter analysis were performed on SUMO.

The dynamic traffic assignment of Sioux Falls, Chicago, and Anaheim runs on SUMO：

![image](https://github.com/user-attachments/assets/3f44c45b-6287-460b-9982-dff2a55d3c19)

![image](https://github.com/user-attachments/assets/6c50eaa2-8043-425a-a76a-9564d7c9cbaa)

![image](https://github.com/user-attachments/assets/ded8dc81-091b-4233-bcca-0009a941b20c)


## requirements
python 3  
torch >= 1.7  
numpy  
scipy  
argparse  
You can install all the requirements by python3 -m pip install -r requirements.txt.


## Train Commands  
``` python train.py ```


## Contact  
dwu3099@gmail.com
