import os
from torch_geometric.nn import radius_graph

import numpy as np
import torch
from loguru import logger

from torch_geometric.data import Data
from tqdm import tqdm

from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius_graph
from Bio.PDB import PDBParser
from tqdm import tqdm
import math 
import multiprocessing


null_structure = ['Q5I6C7', 'Q05470', 'Q6KC79', 'Q96Q15', 'P36022', 'Q96T58', 'Q9VDW6', 'Q3KNY0', 'Q14315', 'Q7TMY8', 'Q9SMH5', 'Q9VC56', 'Q8WXX0', 'Q01484', 'Q5VT06', 'Q8IZQ1', 'Q9P2D1', 
                  'F8VPN2', 'Q9U943', 'O83276', 'P14217', 'Q868Z9', 'O83774', 'Q61001', 'P98161', 'Q9UKN1', 'P04875', 'P0DTC2', 'P29994', 'Q14789', 'P69332', 'Q9VKA4']


class TMPDataset():
    '''
    change the format of DeepTMHMM and create a dictionary for splitting into train, validation, and test data 
    '''
    def __init__(self,data_path):
        self.data_path = data_path
    
    def change_format_raw_data(self,data_path):
        grouped_data = []
        with open(data_path, 'r') as file:
            lines = file.readlines()
        current_group = []
        for line in lines:
            line = line.strip()
            if len(current_group) == 3:
                grouped_data.append(current_group)
                current_group = []
            current_group.append(line)
        if current_group:
            grouped_data.append(current_group)
        return grouped_data
    
    def get_data(self):
        data_dict = {}
        
        i=0
        grouped_data = self.change_format_raw_data(self.data_path)
        atom_info_corres = AtomInfo()
        
        for group in grouped_data:
            raw_header = group[0]
            header = raw_header.split('|')[0]
            type = raw_header.split('|')[1]
            
            if header.startswith('>'):
                header = header[1:]
            if header not in null_structure:
                i+=1
                seq = group[1] 
                label = group[2]
                
                           
                
                            
                seq_list = [char for char in seq]
                atom_list=[]
                for char in seq_list:
                    if char in atom_info_corres.atoms_label:
                        atom_list.extend(atom_info_corres.atoms_label[char]['atoms'])
                            
                data_dict[header] = {
                        "seq": seq,
                        "raw_label": label.split(),
                        "type": type,
                        "atom_length":len(atom_list),
                        "atom_label": atom_list  
                        }
            else:
                logger.info(header+" doesn't have 3D structures and gotten rid of datasets")    
                
        
        logger.info(f'Before removing the number of data is {len(grouped_data)}')
        logger.info(f'After removing the number of data is  {i}')
     
        
        return data_dict
            
        

            
        



    


class GetAtomPosNum():
    """
    exact infromations from protein structer
    renturn atom_positions and atom_numbers and x (tensor zero)
    
    """

    def __init__(self,data_path):
        self.data_path = data_path

    def element_to_number(self,element):
        periodic_table = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
        return periodic_table.get(element, 0)  

    def parse_pdb(self,file_path):
        parser = PDBParser()
        structure = parser.get_structure('protein', file_path)
    
        atom_positions = []
        atom_numbers = []
    
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coord = atom.get_coord()
                        atom_positions.append(coord)    
                        atom_type = atom.element
                        atom_number = self.element_to_number(atom_type)
                        atom_numbers.append(atom_number)
    
       
        atom_positions = torch.tensor(np.array(atom_positions), dtype=torch.float)
        x = torch.zeros(atom_positions.shape[0]) 
        atom_numbers = torch.tensor(atom_numbers, dtype=torch.long)  
        pdb_atom = structure.get_atoms()   
        atoms = list(pdb_atom)
        atoms_length = len(atoms) #后续的切割准备
        CA_index_list = [i for i in range(len(atoms)) if str(atoms[i]) == "<Atom CA>" ]  
        return atom_positions, atom_numbers,CA_index_list,atoms_length

    def get_all(self):
        atomposnump = {}
        for path in self.data_path:
            file_name_with_extension = os.path.basename(path)
            file_name, _ = os.path.splitext(file_name_with_extension)
            atom_positions, atom_numbers,CA_index_list,atoms_length = self.parse_pdb(path)
            atomposnump[file_name] = {"pos":atom_positions,"num":atom_numbers,"CA_index_list":CA_index_list,'atoms_length':atoms_length}
        return atomposnump
    


class ProcessBatchData():
    """
    create the batch, need to send train, validation, and test data respectively.
    based on the batch, combine them into a big graph and exact important information 
    return 
    one of the data in the batch looks like 
    DataBatch(x=[30410], pos=[30410, 3], batch=[30410], ptr=[12], edge_index=[2, 955751], edge_weight=[955751])
    """
    
    def __init__(self,batch_size,file_paths):
        self.batch_size=batch_size
        self.atom_info = GetAtomPosNum(file_paths)


    
    def batch_generator(self,data_list, batch_size):
        '''
        less than batch size is one batch
        '''
        
        #random.shuffle(data_list)
        num_data = len(data_list)
        start = 0
    
        while start < num_data:
            end = min(start + batch_size, num_data)
            batch = data_list[start:end]
            start = end
            yield batch
    
    def get_process_data(self):
        '''
        based on the 3D structures to get the pieces of information including edge_index,edge_weight etc 
        '''
         # {batch_name:batch_length} for processing labels (predict labels align at real labels )
        max_len = 0
        atomposnump = self.atom_info.get_all()
        name_list = list(set(atomposnump))
        total_batches = math.ceil(len(name_list) / self.batch_size)
        batch_iterator = self.batch_generator(name_list, self.batch_size)
        batch_data_list = []
        batch_name_list=[]
        CA_index_list_all =[]
        atoms_length_all =[]
        

        
        for batch in tqdm(batch_iterator,total=(total_batches), desc="Structure Analysis Using Geometric"):
            batch_pos_data = [atomposnump[batch[num]]['pos'] for num in range(len(batch))]
            batch_num_data = [atomposnump[batch[num]]['num'] for num in range(len(batch))]
            CA_index_list  = [atomposnump[batch[num]]['CA_index_list'] for num in range(len(batch))]
            atoms_length = [atomposnump[batch[num]]['atoms_length'] for num in range(len(batch))]
            batch_data =Batch.from_data_list([Data(pos=batch_pos_data[num], x=batch_num_data[num]) for num in range(len(batch_num_data))])
           
            batch_data.edge_index = radius_graph(batch_data.pos, r=8, max_num_neighbors=32,batch=batch_data.batch)
            
            row, col = batch_data.edge_index
            batch_data.edge_weight = (batch_data.pos[row] - batch_data.pos[col]).norm(dim=-1)
            
            batch_data_list.append(batch_data)
            batch_name_list.append(batch)
            CA_index_list_all.append(CA_index_list)
            atoms_length_all.append(atoms_length )

            
            #look for the max length of batches to set the max of static emmbedding 
            if len(batch_data.x) > max_len:
                max_len = len(batch_data.x)
       
        # CA_index_list是个列表包含着每一个的蛋白质的index_list, 后续的你要把列表切开然后进行比对
        # atoms_length中包含的是选用的batch的蛋白质的，每个蛋白质的原子的长度为了后面的切割
        return batch_data_list,batch_name_list,max_len,CA_index_list_all,atoms_length_all

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
