# CommonsenseQA_RN
A repo for my code on Hybrid RN for CommonsenseQA

In utils whichever type of adjacency matrix for schema graphs you want just rename that file to graph.py and then run 

Requirements:

Use these commands to install the requirements:

pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

pip install torch-geometric

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

To run vanilla gn:

Ensure you have the correct grounding files stores with you

python newprocess_1.py

python gn.py --encoder-bert-base-uncased -mbs 4 -dlr 1e-3
