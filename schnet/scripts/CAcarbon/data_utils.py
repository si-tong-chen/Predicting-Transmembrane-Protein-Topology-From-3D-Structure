import os
import json
import requests
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from loguru import logger
from torch_geometric import transforms as T

from tqdm import tqdm
import random


from base import TMPDataset,ProcessBatchData,AtomInfo

class LengthMismatchError(Exception):
    pass

class DismatchIndexPadRawData():
    '''
    batch_size = 1 
    '''
    def __init__(self,batch_name,data,df_val):
        self.batch_name = batch_name #all of the data not batch size data 
        self.data = data ##data is in the get_process_data from class ProcessBatchData（graph data after processing）
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
                last_one = atom_aa_label_mapping[index-1] # 真实的
                if index < len(real_atom):
                    after_one = atom_aa_label_mapping[index]
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
        

        for i in tqdm(range(len(self.batch_name)),total=(len(self.batch_name)), desc="Finding dismatch and processing ..."):
            real_atom=[]
            dismatch_indices = []
            dismatch_type=[]
            pred_label=np.array(self.data[i].x)

            for j in range(len(self.batch_name[i])):
                atom_label = self.df_val[self.df_val['uniprot_id_low'] ==self.batch_name[i][j]]['atom_label'][0]
                
                real_atom.extend(atom_label)
                
            real_atom_label=np.array(real_atom)
                
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
                dismatch_index_pred['_'.join(map(str, self.batch_name[i]))] = dismatch_indices
                dismatch_index_type['_'.join(map(str, self.batch_name[i]))] = 'pred > real'
        
            if len(pred_label) < len(real_atom_label):#输出的真实的index，真实的比预测的长，按照这个index 减去对应的真实的位置
                n = 0  
                for m in range(len(real_atom_label)):
                    if n < len(pred_label) and pred_label[n] == real_atom_label[m]:
                        n += 1
                    else:
                        dismatch_indices.append(m)
                        dismatch_type.append('pred < real')
                dismatch_index_pred['_'.join(map(str, self.batch_name[i]))] = dismatch_indices
                dismatch_index_type['_'.join(map(str, self.batch_name[i]))] = 'pred < real'
        
            sequence=[]
            labels=[]
            for j in range(len(self.batch_name[i])):
                sequence.extend(list(self.df_val[self.df_val['uniprot_id_low'] == self.batch_name[i][j]]['seq'])[0])
                labels.extend(list(self.df_val[self.df_val['uniprot_id_low'] == self.batch_name[i][j]]['raw_label'])[0][0])

            atom_aa_label_mapping_a = self.atom_aa_label_mapping(sequence,labels)
            atom_aa_label_mapping_b = self.pad_del_realdata(dismatch_indices,dismatch_type,atom_aa_label_mapping_a,real_atom,pred_label)
            
            if len(atom_aa_label_mapping_b) != len(pred_label):
                raise LengthMismatchError("Lengths of raw data and predict label do not match.")
            after_process_rawdata['_'.join(map(str, self.batch_name[i]))] = atom_aa_label_mapping_b
        logger.info(f'Have finshed finding dismatch and processing, after processing not founded dismatch')
        return after_process_rawdata,dismatch_index_pred,dismatch_index_type
    




class ProcessRawData():
    def __init__(
        self,
        path: str,
        raw_data_name: str,
        batch_size: int = 1,
        dataset_name_after_process: str = "tmp", 
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 4,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False
    ) -> None:
        self.dataset_name_after_process = dataset_name_after_process
        self.raw_data_name = raw_data_name
        self.batch_size = batch_size
        self.all_data = {}
        self.root = pathlib.Path(path) /"dataset"/ self.dataset_name_after_process 
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        
        self.pdb_dir = pathlib.Path(path) / "dataset"/ "structures"
        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir, exist_ok=True)
        self.raw_data_dir = pathlib.Path(path) / "dataset"/ self.raw_data_name
        self.output_path_amen_dataset = pathlib.Path(path) / "dataset"/'amend dataset'
        if not os.path.exists(self.output_path_amen_dataset):
            os.makedirs(self.output_path_amen_dataset, exist_ok=True)
        
        
        

    def setup(self, stage: Optional[str] = None):
        for split in {"train", "val", "test"}:
            data = self.parse_dataset(split)
            self.all_data[split] = data
            logger.info("Preprocessing " + split +" data")

    def file_paths_process(self,split: Literal["train", "val", "test"]):
        directory = self.pdb_dir/ f"{split}"
        pdb_names = os.listdir(directory)
        file_paths = [str(directory / name) for name in pdb_names]
        return file_paths

    def split_data(self):
        '''
        split the data into train, validation, test and store in the files
        '''
        processor= TMPDataset(self.raw_data_dir)
        data_dict = processor.get_data()
  
        test_SP_TM={}
        test_TM={}
        test_BETA={}
        global_data = {}
        singal_data = {}
        train_data={}
        val_data={}
        keys = list(data_dict.keys())
        for name in keys:
            if data_dict[name]['type'] == 'BETA':
                test_BETA[name]=data_dict[name]
            if data_dict[name]['type'] == 'TM':
                test_TM[name]=data_dict[name]
            if data_dict[name]['type'] == 'SP+TM':
                test_SP_TM[name]=data_dict[name]
            if data_dict[name]['type'] == 'SIGNAL':
                singal_data[name]=data_dict[name]
            if data_dict[name]['type'] == 'GLOBULAR':
                global_data[name]=data_dict[name]
        

        random.seed(42)
        global_keys = list(global_data.keys())
        random.shuffle(global_keys)
        total_global_samples = len(global_keys)
        train_global_keys = global_keys[:int(total_global_samples*0.8)]
        val_global_keys = global_keys[int(total_global_samples*0.8):]

        signal_keys = list(singal_data.keys())
        random.shuffle(signal_keys)
        total_signal_samples = len(signal_keys)
        train_signal_keys = signal_keys[:int(total_signal_samples*0.8)]
        val_signal_keys = signal_keys[int(total_signal_samples*0.8):]



        for name in train_global_keys:
            train_data[name]= global_data[name]
    
        for name in train_signal_keys:
            train_data[name]= singal_data[name]
        
        for name in val_global_keys:   
            val_data[name]= global_data[name]
        for name in val_signal_keys:
            val_data[name]= singal_data[name]

        logger.info(f'The number of train data is {len(train_data)}')
        logger.info(f'The number of val data is {len(val_data)}')   
        logger.info(f'The number of test data is {len(test_TM)+len(test_BETA)+len(test_SP_TM)}')
    
        
        


        output_path_train = self.root / 'train.json'
        output_path_val = self.root / 'val.json'
        output_path_test_SP_TM = self.root / 'test_SP_TM.json'
        output_path_test_TM = self.root / 'test_TM.json'
        output_path_test_BETA = self.root / 'test_BETA.json'


        with open(output_path_train, 'w') as json_file:
            json.dump(train_data, json_file, indent=4)
        with open(output_path_val, 'w') as json_file:
            json.dump(val_data, json_file, indent=4)

        with open(output_path_test_BETA, 'w') as json_file:
            json.dump(test_BETA, json_file, indent=4)
        with open(output_path_test_TM, 'w') as json_file:
            json.dump(test_TM, json_file, indent=4)
        with open(output_path_test_SP_TM, 'w') as json_file:
            json.dump(test_SP_TM, json_file, indent=4)
        
        logger.info(f"Have splited the data into train, validation and test.")
        





    def parse_dataset(self, split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']) -> pd.DataFrame:
        """
        processing the raw data DeepTMHMM(sequence)
        """

        data = json.load(open(self.root / f"{split}.json", "r"))
        data = pd.DataFrame.from_records(data).T
        data["uniprot_id"] = data.index
        data.columns = [
            "seq",
            "raw_label",
            "protein_type",
            "atom_length",
            "atom_label",
            "uniprot_id",
        ]
        data["uniprot_id_low"] = data["uniprot_id"].str.lower()
        data["seq_length"] = data["seq"].apply(len)

        if (data['uniprot_id_low'] == 'q841a2').any():
            data = data[data['uniprot_id_low'] != 'q841a2']
        if (data['uniprot_id_low'] == 'd6r8x8').any():
            data = data[data['uniprot_id_low'] != 'd6r8x8']


        logger.info(f"Found {len(data)} examples in {split}")
        return data
    

    def get_alphafold_db_pdb(self, protein_id: str, out_path: str) -> bool:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        requestURL = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
        r = requests.get(requestURL)

        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
                return True
        else:
            return False

    def download(self, split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
        """
        download the 3D protein structures from alphafold

        """
        data = self.parse_dataset(split)
        logger.info(f"Downloading {split} data structures ...")

        uniprot_ids = list(data["uniprot_id"].unique())

        to_download = [
            id
            for id in uniprot_ids
            if not os.path.exists(self.pdb_dir/split/ f"{id}.pdb")
        ]
 
        logger.info(f"Downloading {len(to_download)} PDBs...")
        for id in to_download:
            out_path = os.path.join(self.pdb_dir,split, f"{id.lower()}.pdb")
            success = self.get_alphafold_db_pdb(id, out_path)
            if success:
                logger.info(f"Downloaded PDB for {id}")
            else:
                logger.warning(f"Failed to download PDB for {id}")
    
    def amend_dataset(self,split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
        '''
        amend the dataset assembling the methods to creat a big tabel including the all of data information
        The sequence,label and atom break down like (7,M,I) in order to match the predict label and make the real lable
        '''
        file_paths = self.file_paths_process(split)
        df_split = self.parse_dataset(split)
        processor = ProcessBatchData(1,file_paths)
        structure_data,batch_name,max_len,_,_ = processor.get_process_data()
        processor_dismatch = DismatchIndexPadRawData(batch_name,structure_data,df_split)

        after_process_rawdata,dismatch_index_pred,dismatch_index_type=processor_dismatch.match_real_geometric()
        
        # with open(f"{self.output_path_amen_dataset}/{split}.json", 'w') as json_file:
            # json.dump(after_process_rawdata, json_file, indent=4)

        return after_process_rawdata,dismatch_index_pred,dismatch_index_type






    def run(self):
        self.split_data()
        self.setup()
        self.download('train')
        self.download('val')   
        self.download('test_SP_TM')
        self.download('test_TM')
        self.download('test_BETA')
        
      
  
