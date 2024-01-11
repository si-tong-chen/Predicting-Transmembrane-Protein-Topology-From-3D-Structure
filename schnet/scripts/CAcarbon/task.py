
from base import ProcessBatchData,AtomInfo
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

from data_utils import ProcessRawData
from tqdm import tqdm
from loguru import logger
from collections import Counter
import numpy as np
one_hot_encoding = {
        'I': [1, 0, 0, 0, 0, 0],
        'O': [0, 1, 0, 0, 0, 0],
        'P': [0, 0, 1, 0, 0, 0],
        'S': [0, 0, 0, 1, 0, 0],
        'M': [0, 0, 0, 0, 1, 0],
        'B': [0, 0, 0, 0, 0, 1]
    }
label_dict = {'I': 0, 'O': 1, 'P': 2, 'S': 3, 'M': 4, 'B': 5}


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

class CreateDataLabel(ProcessRawData):
    def __init__(self, root,batch_size,raw_data_name):
        super().__init__(root,raw_data_name)
        self.batch_size = batch_size
    
    def analyze_structural_data(self, split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
        """
        from base import  ProcessBatchData
        to create the batch data with graph information to use schnet model 
        """
        file_paths = super().file_paths_process(split)
        processor = ProcessBatchData(self.batch_size, file_paths)
        structure_data, batch_name, max_len,CA_index_list,atoms_length = processor.get_process_data()
        return structure_data, batch_name, max_len,CA_index_list,atoms_length
    
    def creat_one_hot_label(self, batch_map):
        one_hot_encoded_list = []
        for _, _, label in batch_map:
            one_hot_label = one_hot_encoding[label]
            one_hot_encoded_list.append(one_hot_label)

        return one_hot_encoded_list

    def datalabelgenerator(self,split: Literal['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']):
        '''
        create the batch data 
        create the final label same as the data graph after gerometric processing

        '''

        #it will be used in the model script
        logger.info("Amending " + split + 'data')
        after_process_rawdata,dismatch_index_pred,dismatch_index_type = super().amend_dataset(split)
        

        logger.info("Processing " + split +" data")
        model_data, batch_name, max_len,CA_index_list,atoms_length = self.analyze_structural_data(split)
        
        model_lable=[]
        
        logger.info("Processing " + split +" labels")

        df = super().parse_dataset(split)

        all_node_label = []
        # for j in tqdm(range(len(batch_name)),total=(len(batch_name)),desc=f"Processing {split} node label"):
        #     rawlabel_total_list=[]
        #     num_total_list=[]

        #     for name in batch_name[j]:
        #         for i in range(len(df[df['uniprot_id_low'] == name]['raw_label'][0][0])):  
        #             rawlabel_total_list.extend(df[df['uniprot_id_low'] == name]['raw_label'][0][0][i])

        #     for char in rawlabel_total_list:    
        #         value = label_dict[char]
        #         num_total_list.append(value)

        #     all_node_label.append(num_total_list)

        for i in tqdm(range(len(batch_name)),total=(len(batch_name)),desc=f"Processing {split} node label"):
            mask = df['uniprot_id_low'].isin(batch_name[i])
            filtered_df = df[mask]
            rawlabel_total_list =[letter for sublist in filtered_df['raw_label'] for item in sublist for letter in item]
            num_total_list = [label_dict[char] for char in rawlabel_total_list]
            all_node_label.append(num_total_list)

        for i in tqdm(range(len(batch_name)), total=(len(batch_name)),desc=f"Processing {split} atom label"):
            batch_map = []
            for j in range(len(batch_name[i])):
                batch_map.extend(after_process_rawdata[batch_name[i][j]])
            if len(batch_map) != len(model_data[i].x):
                raise LengthMismatchError("The length of the batch is not equal to the length of the data after geometric processing")
            
            one_hot_encoded_list=self.creat_one_hot_label(batch_map)
            model_lable.append(one_hot_encoded_list)
        return model_data,model_lable, max_len,all_node_label,CA_index_list,atoms_length
    
    
    def initialization(self):
        super().run()


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
            seq_total_list.extend(self.df_val[self.df_val['uniprot_id_low'] == name]['seq'][0])
            
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
            length_total +=self.df_val[self.df_val['uniprot_id_low'] == name]['seq_length'][0]

        if len(pred_seq) != length_total:
            raise LengthMismatchError("The length of the batch is not equal to the length of the model outcome after processing")

        return pred_seq
    
    
    







        
        