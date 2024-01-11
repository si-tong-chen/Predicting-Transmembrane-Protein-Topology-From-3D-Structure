import os
import json
import requests
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    List,
)
import pandas as pd
from loguru import logger
import  pickle
import torch
from loguru import logger
import numpy as np

from Bio.PDB import PDBParser


null_structure = ['Q5I6C7', 'Q05470', 'Q6KC79', 'Q96Q15', 'P36022', 'Q96T58', 'Q9VDW6', 'Q3KNY0', 'Q14315', 'Q7TMY8', 'Q9SMH5', 'Q9VC56', 'Q8WXX0', 'Q01484', 'Q5VT06', 'Q8IZQ1', 'Q9P2D1', 
                  'F8VPN2', 'Q9U943', 'O83276', 'P14217', 'Q868Z9', 'O83774', 'Q61001', 'P98161', 'Q9UKN1', 'P04875', 'P0DTC2', 'P29994', 'Q14789', 'P69332', 'Q9VKA4']

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
class GetAtomPosNum():
    """
    exact infromations from protein structer
    renturn atom_positions and atom_numbers and x (tensor zero)
    保留的
    
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

        atom_numbers = torch.tensor(atom_numbers, dtype=torch.long) 
        pdb_atom = structure.get_atoms()   
        atoms = list(pdb_atom)
        CA_index_list = [i for i in range(len(atoms)) if str(atoms[i]) == "<Atom CA>" ]
        return atom_positions, atom_numbers, CA_index_list

    def get_all(self):
        atomposnump = {}
        for path in self.data_path:
            file_name_with_extension = os.path.basename(path)
            file_name, _ = os.path.splitext(file_name_with_extension)
            atom_positions, atom_numbers,CA_index_list = self.parse_pdb(path)
            atomposnump[file_name] = {"pos":atom_positions,"num":atom_numbers,"CA_index_list":CA_index_list}
        return atomposnump

class ChangeFormatTMPDataset():
    '''
    change the format of DeepTMHMM and create a dictionary for splitting into train, validation, and test data 

    '''
    def __init__(self,data_path):
        self.data_path = data_path
    
    def change_format_raw_data(self,data_path)->List[str]:
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
    
    def get_data(self) -> Dict[str, Any]:
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
    

class ProcessRawData():
    ''''
    process the raw data and create a dictionary for splitting into train, validation, and test data
    
    '''
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
        # self.dataset_name_after_process = dataset_name_after_process
        self.raw_data_name = raw_data_name
        self.batch_size = batch_size
        self.all_data = {}
        self.root = pathlib.Path(path) /"data"/'processed'/ "dict"
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        
        self.pdb_dir = pathlib.Path(path) / "data"/ 'processed'/"structures"
        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir, exist_ok=True)
        self.raw_data_dir = pathlib.Path(path) / "data"/ "raw"/self.raw_data_name
        self.partitions_dir = pathlib.Path(path) / "data"/ "raw"/'DeepTMHMM.partitions.json'
        self.parsedata_pd = pathlib.Path(path) /"data"/ 'processed'/'dataframe'
        if not os.path.exists(self.parsedata_pd):
            os.makedirs(self.parsedata_pd, exist_ok=True)


    def setup(self, stage: Optional[str] = None):
        for split in {'cv0','cv1','cv2','cv3','cv4'}:
            data = self.parse_dataset(split)
            self.all_data[split] = data
            logger.info("Preprocessing " + split +" data")

    def file_paths_process(self,split: Literal['cv0','cv1','cv2','cv3','cv4']):
        directory = self.pdb_dir/ f"{split}"
        pdb_names = os.listdir(directory)
        file_paths = [str(directory / name) for name in pdb_names]
        return file_paths

    def split_data(self):
        '''
        split the data into different parts and store in the files
        '''
        processor= ChangeFormatTMPDataset(self.raw_data_dir)
        data_dict = processor.get_data()
        null_structure = ['Q5I6C7', 'Q05470', 'Q6KC79', 'Q96Q15', 'P36022', 'Q96T58', 'Q9VDW6', 'Q3KNY0', 'Q14315', 'Q7TMY8', 'Q9SMH5', 'Q9VC56', 'Q8WXX0', 'Q01484', 'Q5VT06', 'Q8IZQ1', 'Q9P2D1', 
                  'F8VPN2', 'Q9U943', 'O83276', 'P14217', 'Q868Z9', 'O83774', 'Q61001', 'P98161', 'Q9UKN1', 'P04875', 'P0DTC2', 'P29994', 'Q14789', 'P69332', 'Q9VKA4']


        f = open(self.partitions_dir)
        cv_data = json.load(f)
        name_list = ['cv0','cv1','cv2','cv3','cv4']


        for name in name_list:
      
            cv0 = cv_data[name]
            cv0_name_list = [cv0[i]['id'] for i in range(len(cv0))]
            cv_name_list = [item for item in cv0_name_list if item not in null_structure]

            cv0_data= {name: data_dict[name] for name in cv_name_list}
            output_path_train = self.root /f"{name}.json"
            with open(output_path_train, 'w') as json_file:
                json.dump(cv0_data, json_file, indent=4)


    def parse_dataset(self, split: Literal['cv0','cv1','cv2','cv3','cv4']) -> pd.DataFrame:
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

        with open(self.parsedata_pd/f"{split}.pickle", 'wb') as file:
            pickle.dump(data, file)

    

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

    def download(self, split: Literal['cv0','cv1','cv2','cv3','cv4']):
        """
        download the 3D protein structures from alphafold

        """

        with open(self.parsedata_pd / f"{split}.pickle", 'rb') as file:
            data = pickle.load(file)


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

    def run(self):
        name_list=['cv0','cv1','cv2','cv3','cv4']
        self.split_data()
        for name in name_list:
            self.parse_dataset(name) 
            # self.download(name)






class ParseStructure():
    '''
    deal with the download pdb and store them
    '''
    def __init__(self,
                 path: str,):

        self.output_path_parse_structure_dataset = pathlib.Path(path) / "data"/'processed'/"parse sturcture dataset"
        self.pdb_dir = pathlib.Path(path) / "data"/'processed'/ "structures"
        if not os.path.exists(self.output_path_parse_structure_dataset):
            os.makedirs(self.output_path_parse_structure_dataset, exist_ok=True)


    def file_paths_process(self,split: Literal['cv0','cv1','cv2','cv3','cv4']):
        directory = self.pdb_dir/ f"{split}"
        pdb_names = os.listdir(directory)
        file_paths = [str(directory / name) for name in pdb_names]
        return file_paths
    
    def store_strcture_data_after_parse(self,split: Literal['cv0','cv1','cv2','cv3','cv4']):
        file_paths = self.file_paths_process(split)
        processor = GetAtomPosNum(file_paths)
        atomposnump = processor.get_all()

        with open(self.output_path_parse_structure_dataset/f"{split}.pickle", 'wb') as file:
            pickle.dump(atomposnump, file)

    def run(self):
        # this is store the structer infromation after parsing, run this can get all we need 
        name_list=['cv0','cv1','cv2','cv3','cv4']
        [self.store_strcture_data_after_parse(split) for split in name_list]


if __name__ == "__main__":
    path = os.getcwd()
    processor = ProcessRawData(path,raw_data_name='DeepTMHMM.3line')
    processor.run()
    processor = ParseStructure(path)
    processor.run()
