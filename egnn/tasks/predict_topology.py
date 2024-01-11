from data.make_dataset import ProcessRawData
from torch.utils.data import Dataset,DataLoader
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data, Batch

from typing import (
    Literal,
)
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import pathlib
import torch
from torch.utils.data import DataLoader
import pickle
from collections import Counter
import numpy as np

from torch import Tensor
one_hot_encoding = {
        'I': [1, 0, 0, 0, 0, 0],
        'O': [0, 1, 0, 0, 0, 0],
        'P': [0, 0, 1, 0, 0, 0],
        'S': [0, 0, 0, 1, 0, 0],
        'M': [0, 0, 0, 0, 1, 0],
        'B': [0, 0, 0, 0, 0, 1]
    }
label_dict = {'I': 0, 'O': 1, 'P': 2, 'S': 3, 'M': 4, 'B': 5}
class AtomInfo:
    def __init__(self):
        self.atoms_label = {'M':{'len':8,'atoms':[7, 6, 6, 6, 8, 6, 16, 6]},
              'L':{'len':8,'atoms':[7, 6, 6, 6, 8, 6, 6, 6]},
              'N':{'len':8,'atoms':[7, 6, 6, 6, 8, 6, 7, 8]},
              'A':{'len':5,'atoms':[7, 6, 6, 6, 8]},
              'S':{'len':6,'atoms':[7, 6, 6, 6, 8, 8]},
              'G':{'len':4,'atoms':[7,6,6,8]},
              'H':{'len':10,'atoms':[7, 6, 6, 6, 8, 6, 6, 7, 6, 7]},
              'K':{'len':9,'atoms':[7, 6, 6, 6, 8, 6, 6, 6, 7]},
              'I':{'len':8,'atoms':[7, 6, 6, 6, 8, 6, 6, 6]},
              'T':{'len':7,'atoms':[7, 6, 6, 6, 8, 6, 8]},
              'R':{'len':11,'atoms':[7, 6, 6, 6, 8, 6, 6, 7, 7, 7, 6]},
              'F':{'len':11,'atoms':[7, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6]},
              'V':{'len':7,'atoms':[7, 6, 6, 6, 8, 6, 6]},
              'E':{'len':9,'atoms':[7, 6, 6, 6, 8, 6, 6, 8, 8]},
              'P':{'len':7,'atoms':[7, 6, 6, 6, 8, 6, 6]},
              'Q':{'len':9,'atoms':[7, 6, 6, 6, 8, 6, 6, 7, 8]},
              'Y':{'len':12,'atoms':[7, 6, 6, 6, 8, 6, 6, 6, 6, 6, 8, 6]},
              'D':{'len':8,'atoms':[7, 6, 6, 6, 8, 6, 8, 8]},
              'W':{'len':14,'atoms':[7, 6, 6, 6, 8, 6, 6, 6, 6, 6, 7, 6, 6, 6]},
              'C':{'len':6,'atoms':[7, 6, 6, 6, 8, 16]},
              'O':{'len':8,'atoms':[7,6,6,6,6,6,8,8]},
              'U':{'len':8,'atoms':[7,6,6,6,6,6,8,8]}}

class LengthMismatchError(Exception):
    pass

class DismatchIndexPadRawData():
    '''
    batch_size = 1 留着 
    '''
    def __init__(self,batch_name,data,df_val):
        self.batch_name = batch_name 
        self.data = data 
        self.df_val = df_val 

    
    def atom_aa_label_mapping(self,sequence, labels):
        '''
        Create a list where each element is a tuple of (atom, amino acid, position label)
        '''
        atom_info=AtomInfo()
        atom_aa_label_mapping = []
        for aa, label in zip(sequence, labels):
            for atom in atom_info.atoms_label[aa]['atoms']:
                atom_aa_label_mapping.append((atom, aa, label))
        
        return atom_aa_label_mapping

    def pad_del_realdata(self,dismatch_indices,dismatch_type,atom_aa_label_mapping,real_atom,pred_label):
        '''
        adding or delete elements in the real data in order to have the same size between predict and real label
        '''
        for i,index in enumerate(dismatch_indices):
            if dismatch_type[i] == 'pred > real':
                # 首先拿到index-1 和 indx+1 的值 如果index+1 没有的话就等于 = index-1的值
                #index+1 的值 是否为7 是7的话 插入（pred_label[index]，index上一个的氨基酸和标签）
                #。                 不是7的话。看pred_index 是否为7 是7 插入（pred_label[index]，inde下一个的氨基酸和标签）
                #                                               不是7 插入（pred_label[index]，inde上一个的氨基酸和标签）
                last_one = atom_aa_label_mapping[index-1]
                if index+1 <len(real_atom):
                    after_one = atom_aa_label_mapping[index+1]
                else:
                    after_one = last_one
        
                if after_one[0]==7:
                     
                    atom_aa_label_mapping.insert(index,(pred_label[index],last_one[1],last_one[2]))
                else:
                    if pred_label[index] == 7:
                        atom_aa_label_mapping.insert(index,(pred_label[index],after_one[1],after_one[2]))
                    else:
                        atom_aa_label_mapping.insert(index,(pred_label[index],last_one[1],last_one[2]))
        
            else:
                pass
    
            
        return atom_aa_label_mapping
        
    
    def match_real_geometric(self):   
        dismatch_index_pred ={}
        dismatch_index_type={}
        after_process_rawdata={}
        

        for name in self.data.keys():
            real_atom=[]
            dismatch_indices = []
            dismatch_type=[]
            pred_label = np.array(self.data[name].x)
    

            
            atom_label = self.df_val[self.df_val['uniprot_id_low'] == name]['atom_label']
                
            real_atom.extend(atom_label)
                
            real_atom_label=np.array(real_atom[0])
                
            if len(pred_label) > len(real_atom_label): #输出的index是真实的index，真实的比预测的短，找到预测的index（其他的算法） 加入到真实的index位置
                n = 0  
                m = 0  
                while n < len(pred_label):
                    if m >= len(real_atom_label) or pred_label[n] != real_atom_label[m]:
                        
                        dismatch_indices.append(n)  
                        dismatch_type.append('pred > real') 
                        n += 1  
                    else: 
                       
                        n += 1
                        m += 1
                dismatch_index_pred[name] = dismatch_indices
                dismatch_index_type[name] = 'pred > real'
        
            if len(pred_label) < len(real_atom_label):#输出的真实的index，真实的比预测的长，按照这个index 减去对应的真实的位置
                n = 0  
                for m in range(len(real_atom_label)):
                    if n < len(pred_label) and pred_label[n] == real_atom_label[m]:
                        n += 1
                    else:
                        dismatch_indices.append(m)
                        dismatch_type.append('pred < real')
                dismatch_index_pred[name] = dismatch_indices
                dismatch_index_type[name] = 'pred < real'
        
            sequence=[]
            labels=[]
            
            sequence.extend(list(self.df_val[self.df_val['uniprot_id_low'] == name]['seq'])[0])
            labels.extend(list(self.df_val[self.df_val['uniprot_id_low'] == name]['raw_label'])[0][0])

            atom_aa_label_mapping_a = self.atom_aa_label_mapping(sequence,labels)
            atom_aa_label_mapping_b = self.pad_del_realdata(dismatch_indices,dismatch_type,atom_aa_label_mapping_a,real_atom,pred_label)
            
            if len(atom_aa_label_mapping_b) != len(pred_label):
                raise LengthMismatchError("Lengths of raw data and predict label do not match.")
            after_process_rawdata[name] = atom_aa_label_mapping_b
       

        return after_process_rawdata,dismatch_index_pred,dismatch_index_type
    

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class ProcessBatch():
    def __init__(self):
        pass

    def _normalize(self,tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Safely normalize a Tensor. Adapted from:
        https://github.com/drorlab/gvp-pytorch.

        :param tensor: Tensor of any shape.
        :type tensor: Tensor
        :param dim: The dimension over which to normalize the input Tensor.
        :type dim: int, optional
        :return: The normalized Tensor.
        :rtype: torch.Tensor
        """
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
        )
    
    def add_edge_features(self,x):

        vector_edge_features = []
        E_vectors = x.pos[x.edge_index[0]] - x.pos[x.edge_index[1]]
        vector_edge_features.append(self._normalize(E_vectors).unsqueeze(-2))
        x.edge_vector_attr = torch.cat(vector_edge_features, dim=0)
        return x

    def orientations(self,X, ca_idx: int = 1):
        if X.ndim == 3:
            X = X[:, ca_idx, :]
        forward = self._normalize(X[1:] - X[:-1])
        backward =self._normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)

    def add_node_features(self,x):
        vector_node_features = []
        vector_node_features.append(self.orientations(x.pos))
        x.x_vector_attr = torch.cat(vector_node_features, dim=0)
        return x

    def batchdata(self,data_batch):
        
        batch_pos_data = [data_batch[num]['data']['pos'] for num in range(len(data_batch))]
        batch_num_data=[data_batch[num]['data']['num'] for num in range(len(data_batch))]
        batch_data =Batch.from_data_list([Data(pos=batch_pos_data[num], x=batch_num_data[num]) for num in range(len(batch_num_data))])

        batch_data = batch_data.to('cuda')
        batch_data.edge_index = radius_graph(batch_data.pos, r=8, max_num_neighbors=32,batch=batch_data.batch)
        
        row, col = batch_data.edge_index
        batch_data.edge_weight = (batch_data.pos[row] - batch_data.pos[col]).norm(dim=-1)
        batch_data=self.add_edge_features(batch_data)
        batch_data= self.add_node_features(batch_data)
        
        batch_data = batch_data.to('cpu')
        distance_expansion = GaussianSmearing(0.0, 5.0, num_gaussians=50)
        edge_attr = distance_expansion(batch_data.edge_weight)
        edge_attr = torch.mean(edge_attr, dim=1)
        batch_data.edge_attr =  edge_attr 
        return batch_data


def custom_collate(batch):
    # 检查是否需要处理DataBatch类型的对象
    if isinstance(batch[0], Data):
        return Batch.from_data_list(batch)
    else:
        # 默认情况下，使用PyTorch的默认collate_fn函数处理其他类型的数据
        return torch.utils.data.dataloader.default_collate(batch)
    
def node_accuracy(val_predict_node_label,val_real_node_label):
    ''''
    calculate the accuracy of the node label
    '''
    accuracy_list = [1 if x == y else 0 for x, y in zip(val_predict_node_label, val_real_node_label)]
    correct_count = sum(accuracy_list)
    accuracy = correct_count / len(val_predict_node_label)
    
    return accuracy


class LengthMismatchError(Exception):
    pass
class ValueError(Exception):
    pass



class CreateDataBeforeBatch():
    def __init__(self,
                 path:str):
        self.output_path_parse_structure_dataset = pathlib.Path(path) / "data"/'processed'/"parse sturcture dataset"


    def get_data(self, split: Literal['setup1','setup2','setup3','setup4','setup5']):
        if split == 'setup1':
            train_list =  ['cv0','cv1','cv2']
            val_list = 'cv3'
            test_list = 'cv4'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)
        
        if split == 'setup2':
            train_list =  ['cv1','cv2','cv3']
            val_list = 'cv4'
            test_list = 'cv0'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)
    
        if split == 'setup3':
            train_list =  ['cv2','cv3','cv4']
            val_list = 'cv0'
            test_list = 'cv1'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)

        if split == 'setup4':
            train_list =  ['cv3','cv4','cv0']
            val_list = 'cv1'
            test_list = 'cv2'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)

        if split == 'setup5':
            train_list =  ['cv4','cv0','cv1']
            val_list = 'cv2'
            test_list = 'cv3'

            train_data = {}
            for name in train_list:
                file_path = self.output_path_parse_structure_dataset/f"{name}.pickle"
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    train_data.update(loaded_data)

            val_file_path = self.output_path_parse_structure_dataset/f"{val_list}.pickle"
            with open(val_file_path, 'rb') as file:
                val_data = pickle.load(file)

            test_file_path = self.output_path_parse_structure_dataset/f"{test_list}.pickle"
            with open(test_file_path, 'rb') as file:
                test_data = pickle.load(file)

        return train_data,val_data,test_data


       
        



    



class CreateLable(ProcessRawData):
    def __init__(self,batchname,data_batch,path,raw_data_name):
        super().__init__(path,raw_data_name)
        self.batchname=batchname
        self.data_batch = data_batch
        self.parsedata_pd = pathlib.Path(path) /"data"/ 'processed'/'dataframe'
       
    

    def creat_one_hot_label(self, batch_map):
            one_hot_encoded_list = []
            for _, _, label in batch_map:
                one_hot_label = one_hot_encoding[label]
                one_hot_encoded_list.append(one_hot_label)

            return one_hot_encoded_list
    
    def df_dataset(self,split: Literal['setup1','setup2','setup3','setup4','setup5']):

        '''
        amend the dataset assembling the methods to creat a big tabel including the all of data information
        The sequence,label and atom break down like (7,M,I) in order to match the predict label and make the real lable
        '''
        if split == 'setup1':
            train_list =  ['cv0','cv1','cv2']
            val_list = 'cv3'
            test_list = 'cv4'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{'cv3'}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{'cv4'}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe           
            
        if split == 'setup2':
            train_list =  ['cv1','cv2','cv3']
            val_list = 'cv4'
            test_list = 'cv0'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe  

        if split == 'setup3':
            train_list =  ['cv2','cv3','cv4']
            val_list = 'cv0'
            test_list = 'cv1'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe  
        
        if split == 'setup4':
            train_list =  ['cv3','cv4','cv0']
            val_list = 'cv1'
            test_list = 'cv2'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe  

        if split == 'setup5':
            train_list =  ['cv4','cv0','cv1']
            val_list = 'cv2'
            test_list = 'cv3'

            train_df= pd.DataFrame() 
            for name in train_list:
                file_path = self.parsedata_pd/f"{name}.pickle"

                with open(file_path, 'rb') as file:
                    df = pickle.load(file) #这里是dataframe
                train_df = pd.concat([train_df, df], axis=0, ignore_index=True)

            file_path = self.parsedata_pd/f"{val_list}.pickle" 
            with open(file_path, 'rb') as file:
                val_df = pickle.load(file) #这里是dataframe
               
            file_path = self.parsedata_pd/f"{test_list}.pickle" 
            with open(file_path, 'rb') as file:
                test_df = pickle.load(file) #这里是dataframe 

        return  train_df,val_df,test_df







    def amend_dataset(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):
        train_df,val_df,test_df = self.df_dataset(split)
        if subset == 'train':
            data_dict_model = {}
            for num in range(len(self.data_batch)):
                data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
                data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
                row, col = data.edge_index
                data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                name = self.data_batch[num]['name']
                data_dict_model[name]= data 

            precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,train_df)
            after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()
        
        if subset == 'val':
            data_dict_model = {}
            for num in range(len(self.data_batch)):
                data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
                data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
                row, col = data.edge_index
                data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                name = self.data_batch[num]['name']
                data_dict_model[name]= data 

            precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,val_df)
            after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()

        if subset == 'test':
            data_dict_model = {}
            for num in range(len(self.data_batch)):
                data =Batch.from_data_list([Data(pos=self.data_batch[num]['data']['pos'], x=self.data_batch[num]['data']['num'])])
                data.edge_index = radius_graph(data.pos, r=8, max_num_neighbors=32,batch=data.batch)
                row, col = data.edge_index
                data.edge_weight = (data.pos[row] - data.pos[col]).norm(dim=-1)
                name = self.data_batch[num]['name']
                data_dict_model[name]= data 

            precossor = DismatchIndexPadRawData(self.batchname,data_dict_model,test_df)
            after_process_rawdata,dismatch_index_pred,dismatch_index_type= precossor.match_real_geometric()

  
        return after_process_rawdata,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df
    


    def createatomlevellable(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):
  
        after_process_rawdata,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df = self.amend_dataset(split,subset)
    

        atom_level_label_dict = {}
        for name in self.batchname:
            one_hot_encoded_list=self.creat_one_hot_label(after_process_rawdata[name])
            sequence = torch.argmax(torch.tensor(one_hot_encoded_list), dim=1)
            atom_level_label_dict[name]=sequence
        return atom_level_label_dict,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df
    

    
    def creatresiduallevellabel(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):
        train_df,val_df,test_df = self.df_dataset(split)
        if subset == 'train':       
            real_node_level_label_dict={}
            for name in self.batchname:
                filtered_df= train_df[train_df['uniprot_id_low'] == name]
                rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
                num_total_list = [label_dict[char] for char in rawlabel_total_list]

                real_node_level_label_dict[name] = num_total_list

        if subset == 'val':       
            real_node_level_label_dict={}
            for name in self.batchname:
                filtered_df= val_df[val_df['uniprot_id_low'] == name]
                rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
                num_total_list = [label_dict[char] for char in rawlabel_total_list]

                real_node_level_label_dict[name] = num_total_list
        if subset == 'test':       
            real_node_level_label_dict={}
            for name in self.batchname:
                filtered_df= test_df[test_df['uniprot_id_low'] == name]
                rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
                num_total_list = [label_dict[char] for char in rawlabel_total_list]

                real_node_level_label_dict[name] = num_total_list

        return real_node_level_label_dict
    
    def labeldispatcher(self,split: Literal['setup1','setup2','setup3','setup4','setup5'],subset: Literal['train', 'val', 'test'] = 'train'):

        atom_level_label_dict,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df=self.createatomlevellable(split,subset)
        real_node_level_label_dict = self.creatresiduallevellabel(split,subset)
        
        return atom_level_label_dict,real_node_level_label_dict,dismatch_index_pred,dismatch_index_type,train_df,val_df,test_df




class MapAtomNode():
    def __init__(self, predicted,val_batchname,val_dismatch_index_pred,val_dismatch_index_type,df_val):
       
        self.predicted = predicted
        self.val_batchname = val_batchname
        self.val_dismatch_index_pred = val_dismatch_index_pred  
        self.val_dismatch_index_type = val_dismatch_index_type
        self.df_val = df_val
    

    def find_most_frequent_elements(self,lst):
        if not lst:
            return []

        count = Counter(lst)
        max_occurrence = max(count.values())
        most_frequent = [num for num, occ in count.items() if occ == max_occurrence]

        return most_frequent
    

    
    
    def prcoess_predit_label(self):
        '''
        get rid of the wrong index from the predicted label called 'predicted_list'
        and create the length respondding to the node label called 'consecutive_lengths'
        '''
        atom_info=AtomInfo()

        delete_index=0
        seq_total_list = []
        consecutive_lengths=[]
        predicted_list=list(np.array(self.predicted)) # 
        

        for name in self.val_batchname:
            if self.val_dismatch_index_type[name] == 'pred > real':
                index = self.val_dismatch_index_pred[name]
                delete_index +=index[0]
                if 0 <= delete_index < len(predicted_list):
                    del predicted_list[delete_index]
                else: 
                    raise ValueError('index out of range')
            seq_total_list.extend(self.df_val[self.df_val['uniprot_id_low'] == name]['seq'].iloc[0])
            
        for char in seq_total_list:
            consecutive_lengths.append(atom_info.atoms_label[char]['len'])
        
        return predicted_list,consecutive_lengths
    
    def map_atom_node(self):
        '''
        map the atom label to the node label
        '''
        pred_seq = []
        first_one = 0
        last_one = 0
        length_total = 0
        predicted_list,consecutive_lengths = self.prcoess_predit_label()
        for num in consecutive_lengths:
            last_one += num
            list_part = predicted_list[first_one:last_one]
            most_common_one =self.find_most_frequent_elements(list_part)
            pred_seq.append(most_common_one[0])
            first_one = last_one
        for name in self.val_batchname:
            length_total +=self.df_val[self.df_val['uniprot_id_low'] == name]['seq_length'].iloc[0]

        if len(pred_seq) != length_total:
            raise LengthMismatchError("The length of the batch is not equal to the length of the model outcome after processing")

        return pred_seq
    

class TMPDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]     
        data = self.data_dict[key]
        
        return {'name': key,
                'data':data}



class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=1):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)







class LableTogther():
    def __init__(self,cfg):
        self.cfg = cfg

    def processinglabel(self,):
        path= self.cfg['path']
        batch_size = self.cfg['batch_size']
        setup = self.cfg['setup'] # choose crossvalidation (total 5)
        processsor= CreateDataBeforeBatch(path)
        train_data_dict_before_batch,val_data_dict_before_batch,test_data_dict_before_batch=processsor.get_data(setup)

        ## dataloader for processing label 
        train_dataset = TMPDataset(train_data_dict_before_batch)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=lambda x: x,pin_memory=True)

        val_dataset = TMPDataset(val_data_dict_before_batch)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,collate_fn=lambda x: x,pin_memory=True)

        test_dataset = TMPDataset(test_data_dict_before_batch)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,collate_fn=lambda x: x,pin_memory=True)

        return train_data_loader,val_data_loader,test_data_loader
    
    def processinglabel2(self,):
        train_residual_level_label={}
        train_atom_levl_label = {}
        train_dismatch_index_pred ={}
        train_dismatch_index_type ={}
        train_data_loader,val_data_loader,test_data_loader = self. processinglabel()
        path= self.cfg['path']
        file_name = self.cfg['file_name']
        setup = self.cfg['setup']


        for data_batch in train_data_loader:
            batchname=[data_batch[num]['name'] for num in range(len(data_batch))]

            labelprocessor=CreateLable(batchname,data_batch,path,file_name)
            atom_level_label_dict,redidual_level_label_dict,dismatch_index_pred,dismatch_index_type,df_train,_,_=labelprocessor.labeldispatcher(setup,subset='train')

            train_atom_levl_label.update(atom_level_label_dict) 
            train_residual_level_label.update(redidual_level_label_dict) 
            train_dismatch_index_pred.update(dismatch_index_pred)
            train_dismatch_index_type.update(dismatch_index_type)

        val_residual_level_label={}
        val_atom_levl_label = {}
        val_dismatch_index_pred ={}
        val_dismatch_index_type ={}
        for data_batch in val_data_loader:
            batchname=[data_batch[num]['name'] for num in range(len(data_batch))]

            labelprocessor=CreateLable(batchname,data_batch,path,file_name)
            atom_level_label_dict,redidual_level_label_dict,dismatch_index_pred,dismatch_index_type,_,df_val,_=labelprocessor.labeldispatcher(setup,subset='val')
            val_atom_levl_label.update(atom_level_label_dict) 
            val_residual_level_label.update(redidual_level_label_dict) 
            val_dismatch_index_pred.update(dismatch_index_pred)
            val_dismatch_index_type.update(dismatch_index_type)
        

        return train_residual_level_label,train_atom_levl_label,train_dismatch_index_pred,train_dismatch_index_type,val_residual_level_label,val_atom_levl_label,val_dismatch_index_pred,val_dismatch_index_type,df_train,df_val








