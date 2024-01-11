## Predicting Transmembrane Protein Topology From 3D Structure

### Introduction
 Our study introduces a Graph Neural Network (GNN)-based method that uses 3D structural information obtained from the well-known Alphafold model developed by DeepMind. 

 We have completed three sub-projects, each based on different GNN models (Schnet, EGNN, GCPNet), and conducted both horizontal and vertical comparisons. For details, please refer to the [notebook](https://github.com/si-tong-chen/Predicting-Transmembrane-Protein-Topology-From-3D-Structure/blob/main/notebook/Predicting%20transmembrance%20protein%20topology%20from%203D%20structure.ipynb), [report](https://github.com/si-tong-chen/Predicting-Transmembrane-Protein-Topology-From-3D-Structure/blob/main/reports/report.pdf), and [poster](https://github.com/si-tong-chen/Predicting-Transmembrane-Protein-Topology-From-3D-Structure/blob/main/poster/poster.pdf).


 ### Getting Started

 #### Prerequisites
 - Ensure you have Python installed on your system. You can download Python from python.org.(Python Version >=3.9)
 - [Optional] Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
#### Installing the Project
The project can be installed directly using pip which will also handle the installation of all required dependencies. 
```bash
cd gcpnet/egnn/schnet
pip install .
```
This command tells **pip** to install the current package (denoted by .) along with its dependencies. The dependencies and any other installation instructions are defined in the **pyproject.toml** file.


#### Running 

##### EGNN, GCPNet project 
```bash
cd egnn/gcpnet
make data  # processing the raw data and download the 3D structure 
make train # training the model 
make test  # test and evaluate the model 
```
##### Schnet project 
```bash
cd schnet
make data  # processing the raw data and download the 3D structure 
```
please follow the Schnet part in the notebook for training, test and evaluation

### Contributors
Thanks to the following people who have contributed to this project:
- [Xiaopeng Mao](https://github.com/XiaoM96) 

### License
This project uses the following license: MIT.










