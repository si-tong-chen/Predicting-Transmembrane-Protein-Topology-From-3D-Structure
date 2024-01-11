#!/usr/bin/env python
# coding: utf-8

# In[1]:


from task import CreateDataLabel,MapAtomNode,node_accuracy
from schnet import SchNetModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import seaborn as sns


# In[2]:


##['train', 'val', 'test_SP_TM', 'test_TM', 'test_BETA']
batch_size = 1
raw_data_name = "DeepTMHMM.3line"
path ='/work3/s194408/Project/'
processor = CreateDataLabel(path,batch_size =batch_size,raw_data_name=raw_data_name)
# processor.initialization()# split and download trian/val/test just once

train_data,train_lable, train_batchname, train_max_len,train_dismatch_index_pred,train_dismatch_index_type,train_real_node_label,df_train = processor.datalabelgenerator('train')

val_data,val_lable, val_batchname, val_max_len,val_dismatch_index_pred,val_dismatch_index_type,val_real_node_label,df_val = processor.datalabelgenerator('val')

test_SP_TM_data,test_SP_TM_lable, test_SP_TM_batchname, test_SP_TM_max_len,test_SP_TM_dismatch_index_pred,test_SP_TM_dismatch_index_type,test_SP_TM_real_node_label,df_test_SP_TM = processor.datalabelgenerator('test_SP_TM')

test_TM_data,test_TM_lable, test_TM_batchname, test_TM_max_len,test_TM_dismatch_index_pred,test_TM_dismatch_index_type,test_TM_real_node_label,df_test_TM = processor.datalabelgenerator('test_TM')

test_BETA_data,test_BETA_lable, test_BETA_batchname, test_BETA_max_len,test_BETA_dismatch_index_pred,test_BETA_dismatch_index_type,test_BETA_real_node_label,df_test_BETA = processor.datalabelgenerator('test_BETA')


# In[3]:


import json

# Opening JSON file
f = open('/work3/s194408/Project/dataset/tmp/DeepTMHMM.partitions.json')

# returns JSON object as
# a dictionary
cv_data = json.load(f)

cv0 = cv_data['cv0']
cv1 = cv_data['cv1']
cv2 = cv_data['cv2']
cv3 = cv_data['cv3']
cv4 = cv_data['cv4']


# In[8]:


import pandas as pd

# Group the data together
total_data = train_data.copy()
total_label = train_lable.copy()
total_batchname = train_batchname.copy()
total_max_len = train_max_len + val_max_len + test_SP_TM_max_len + test_TM_max_len + test_BETA_max_len
total_dismatch_index_pred = train_dismatch_index_pred.copy()
total_dismatch_index_type = train_dismatch_index_type.copy()
total_real_node_label = train_real_node_label.copy()

frames = [df_train, df_val, df_test_SP_TM, df_test_TM, df_test_BETA]
total_df = pd.concat(frames)

#train_data,train_lable, train_batchname, train_max_len,train_dismatch_index_pred,train_dismatch_index_type,train_real_node_label,df_train 
#test_SP_TM_data,test_SP_TM_lable, test_SP_TM_batchname, test_SP_TM_max_len,test_SP_TM_dismatch_index_pred,test_SP_TM_dismatch_index_type,test_SP_TM_real_node_label,test_SP_TM_val
#test_TM_data,test_TM_lable, test_TM_batchname, test_TM_max_len,test_TM_dismatch_index_pred,test_TM_dismatch_index_type,test_TM_real_node_label,test_TM_val 
#test_BETA_data,test_BETA_lable, test_BETA_batchname, test_BETA_max_len,test_BETA_dismatch_index_pred,test_BETA_dismatch_index_type,test_BETA_real_node_label,test_BETA_val
for i in range(0, len(val_data)):
  total_data.append(val_data[i])
  total_label.append(val_lable[i])
  total_batchname.append(val_batchname[i])
  total_dismatch_index_pred[list(val_dismatch_index_pred)[i]] = list(val_dismatch_index_pred.values())[i]
  total_dismatch_index_type[list(val_dismatch_index_type)[i]] = list(val_dismatch_index_type.values())[i]
  total_real_node_label.append(val_real_node_label[i])


for i in range(0, len(test_SP_TM_data)):
  total_data.append(test_SP_TM_data[i])
  total_label.append(test_SP_TM_lable[i])
  total_batchname.append(test_SP_TM_batchname[i])
  total_dismatch_index_pred[list(test_SP_TM_dismatch_index_pred)[i]] = list(test_SP_TM_dismatch_index_pred.values())[i]
  total_dismatch_index_type[list(test_SP_TM_dismatch_index_type)[i]] = list(test_SP_TM_dismatch_index_type.values())[i]
  total_real_node_label.append(test_SP_TM_real_node_label[i])


for i in range(0, len(test_TM_data)):
  total_data.append(test_TM_data[i])
  total_label.append(test_TM_lable[i])
  total_batchname.append(test_TM_batchname[i])
  total_dismatch_index_pred[list(test_TM_dismatch_index_pred)[i]] = list(test_TM_dismatch_index_pred.values())[i]
  total_dismatch_index_type[list(test_TM_dismatch_index_type)[i]] = list(test_TM_dismatch_index_type.values())[i]
  total_real_node_label.append(test_TM_real_node_label[i])


for i in range(0, len(test_BETA_data)):
  total_data.append(test_BETA_data[i])
  total_label.append(test_BETA_lable[i])
  total_batchname.append(test_BETA_batchname[i])
  total_dismatch_index_pred[list(test_BETA_dismatch_index_pred)[i]] = list(test_BETA_dismatch_index_pred.values())[i]
  total_dismatch_index_type[list(test_BETA_dismatch_index_type)[i]] = list(test_BETA_dismatch_index_type.values())[i]
  total_real_node_label.append(test_BETA_real_node_label[i])



# In[20]:


# For cv0
cv0_data = []
cv0_label = []
cv0_batchname = []
cv0_dismatch_index_pred = {}
cv0_dismatch_index_type = {}
cv0_real_node_label = []
cv0_index = []
cv0_df = pd.DataFrame(columns=total_df.columns)

# The batch names have the same order as the data
for i in range(0, len(cv0)):
  try:
    #cv0_index.append(total_batchname.index([cv0[i]['id'].lower()]))
    cv0_batchname.append([total_df.index[total_df['uniprot_id_low'] == cv0[i]['id'].lower()].tolist()[0].lower()])
  except:
    pass
    #print(cv0[i]['id'].lower() + "is missing for cv0.")


# Find index for the found cv0 proteins
# Note that the labels are aligned with the batch names
for i in range(0, len(cv0)):
  try:
    cv0_index.append(total_batchname.index([cv0[i]['id'].lower()]))
  except:
    pass


# gather the data for cv0
for i in range(0, len(cv0_index)):
  cv0_data.append(total_data[cv0_index[i]])
  cv0_label.append(total_label[cv0_index[i]])
  cv0_dismatch_index_pred[list(total_dismatch_index_pred)[cv0_index[i]]] = list(total_dismatch_index_pred.values())[cv0_index[i]]
  cv0_dismatch_index_type[list(total_dismatch_index_type)[cv0_index[i]]] = list(total_dismatch_index_type.values())[cv0_index[i]]
  cv0_real_node_label.append(total_real_node_label[cv0_index[i]])

  #cv0_df = pd.concat([cv0_df, total_df.loc[[total_df.iloc[cv0_index[i]]["uniprot_id"]]]], ignore_index=False)
  cv0_df = pd.concat([cv0_df, total_df.loc[[cv0_batchname[i][0].upper()]]], ignore_index=False)

# put into a list
cv0_lis = [cv0_data, cv0_label, cv0_batchname, cv0_dismatch_index_pred, cv0_dismatch_index_type, cv0_real_node_label, cv0_df]





# for cv1
cv1_data = []
cv1_label = []
cv1_batchname = []
cv1_dismatch_index_pred = {}
cv1_dismatch_index_type = {}
cv1_real_node_label = []
cv1_index = []
cv1_df = pd.DataFrame(columns=total_df.columns)

# The batch names have the same order as the data
for i in range(0, len(cv1)):
  try:
    cv1_batchname.append([total_df.index[total_df['uniprot_id_low'] == cv1[i]['id'].lower()].tolist()[0].lower()])
  except:
    pass

# Find index for the found cv1 proteins
# Note that the labels are aligned with the batch names
for i in range(0, len(cv1)):
  try:
    cv1_index.append(total_batchname.index([cv1[i]['id'].lower()]))
  except:
    pass
    #print(cv1[i]['id'].lower() + "is missing for cv1.")


# gather the data for cv1
for i in range(0, len(cv1_index)):
  cv1_data.append(total_data[cv1_index[i]])
  cv1_label.append(total_label[cv1_index[i]])
  cv1_dismatch_index_pred[list(total_dismatch_index_pred)[cv1_index[i]]] = list(total_dismatch_index_pred.values())[cv1_index[i]]
  cv1_dismatch_index_type[list(total_dismatch_index_type)[cv1_index[i]]] = list(total_dismatch_index_type.values())[cv1_index[i]]
  cv1_real_node_label.append(total_real_node_label[cv1_index[i]])
  cv1_df = pd.concat([cv1_df, total_df.loc[[cv1_batchname[i][0].upper()]]], ignore_index=False)


# put into a list
cv1_lis = [cv1_data, cv1_label, cv1_batchname, cv1_dismatch_index_pred, cv1_dismatch_index_type, cv1_real_node_label, cv1_df]



# for cv2
cv2_data = []
cv2_label = []
cv2_batchname = []
cv2_dismatch_index_pred = {}
cv2_dismatch_index_type = {}
cv2_real_node_label = []
cv2_index = []
cv2_df = pd.DataFrame(columns=total_df.columns)

# The batch names have the same order as the data
for i in range(0, len(cv2)):
  try:
    cv2_batchname.append([total_df.index[total_df['uniprot_id_low'] == cv2[i]['id'].lower()].tolist()[0].lower()])
  except:
    pass
    #print(cv2[i]['id'].lower() + "is missing for cv2.")


# Find index for the found cv2 proteins
# Note that the labels are aligned with the batch names
for i in range(0, len(cv2)):
  try:
    cv2_index.append(total_batchname.index([cv2[i]['id'].lower()]))
  except:
    pass




# gather the data for cv2
for i in range(0, len(cv2_index)):
  cv2_data.append(total_data[cv2_index[i]])
  cv2_label.append(total_label[cv2_index[i]])
  cv2_dismatch_index_pred[list(total_dismatch_index_pred)[cv2_index[i]]] = list(total_dismatch_index_pred.values())[cv2_index[i]]
  cv2_dismatch_index_type[list(total_dismatch_index_type)[cv2_index[i]]] = list(total_dismatch_index_type.values())[cv2_index[i]]
  cv2_real_node_label.append(total_real_node_label[cv2_index[i]])
  cv2_df = pd.concat([cv2_df, total_df.loc[[cv2_batchname[i][0].upper()]]], ignore_index=False)


# put into a list
cv2_lis = [cv2_data, cv2_label, cv2_batchname, cv2_dismatch_index_pred, cv2_dismatch_index_type, cv2_real_node_label, cv2_df]




# for cv3
cv3_data = []
cv3_label = []
cv3_batchname = []
cv3_dismatch_index_pred = {}
cv3_dismatch_index_type = {}
cv3_real_node_label = []
cv3_index = []
cv3_df = pd.DataFrame(columns=total_df.columns)
# The batch names have the same order as the data
for i in range(0, len(cv3)):
  try:
    cv3_batchname.append([total_df.index[total_df['uniprot_id_low'] == cv3[i]['id'].lower()].tolist()[0].lower()])
  except:
    pass
    #print(cv3[i]['id'].lower() + "is missing for cv3.")


# Find index for the found cv3 proteins
# Note that the labels are aligned with the batch names
for i in range(0, len(cv3)):
  try:
    cv3_index.append(total_batchname.index([cv3[i]['id'].lower()]))
  except:
    pass

# gather the data for cv3
for i in range(0, len(cv3_index)):
  cv3_data.append(total_data[cv3_index[i]])
  cv3_label.append(total_label[cv3_index[i]])
  cv3_dismatch_index_pred[list(total_dismatch_index_pred)[cv3_index[i]]] = list(total_dismatch_index_pred.values())[cv3_index[i]]
  cv3_dismatch_index_type[list(total_dismatch_index_type)[cv3_index[i]]] = list(total_dismatch_index_type.values())[cv3_index[i]]
  cv3_real_node_label.append(total_real_node_label[cv3_index[i]])
  cv3_df = pd.concat([cv3_df, total_df.loc[[cv3_batchname[i][0].upper()]]], ignore_index=False)


# put into a list
cv3_lis = [cv3_data, cv3_label, cv3_batchname, cv3_dismatch_index_pred, cv3_dismatch_index_type, cv3_real_node_label, cv3_df]




# for cv4
cv4_data = []
cv4_label = []
cv4_batchname = []
cv4_dismatch_index_pred = {}
cv4_dismatch_index_type = {}
cv4_real_node_label = []
cv4_index = []
cv4_df = pd.DataFrame(columns=total_df.columns)
# The batch names have the same order as the data
for i in range(0, len(cv4)):
  try:
    cv4_batchname.append([total_df.index[total_df['uniprot_id_low'] == cv4[i]['id'].lower()].tolist()[0].lower()])
  except:
    pass
    #print(cv4[i]['id'].lower() + "is missing for cv4.")


# Find index for the found cv4 proteins
# Note that the labels are aligned with the batch names
for i in range(0, len(cv4)):
  try:
    cv4_index.append(total_batchname.index([cv4[i]['id'].lower()]))
  except:
    pass

# gather the data for cv4
for i in range(0, len(cv4_index)):
  cv4_data.append(total_data[cv4_index[i]])
  cv4_label.append(total_label[cv4_index[i]])
  cv4_dismatch_index_pred[list(total_dismatch_index_pred)[cv4_index[i]]] = list(total_dismatch_index_pred.values())[cv4_index[i]]
  cv4_dismatch_index_type[list(total_dismatch_index_type)[cv4_index[i]]] = list(total_dismatch_index_type.values())[cv4_index[i]]
  cv4_real_node_label.append(total_real_node_label[cv4_index[i]])
  cv4_df = pd.concat([cv4_df, total_df.loc[[cv4_batchname[i][0].upper()]]], ignore_index=False)


# put into a list
cv4_lis = [cv4_data, cv4_label, cv4_batchname, cv4_dismatch_index_pred, cv4_dismatch_index_type, cv4_real_node_label, cv4_df]


# In[26]:


def make_splits(cv0_lis, cv1_lis, cv2_lis, cv3_lis, cv4_lis, setup):

  # for setup 1:
  #cv0, cv1, cv2 for train, cv3 for validation, cv4 for test
  if setup == "setup 1":
  # data format
  #cv0_lis = [cv0_data, cv0_label, cv0_batchname, cv0_dismatch_index_pred, cv0_dismatch_index_type, cv0_real_node_label, cv0_df]

    setup_train_data = cv0_lis[0].copy()
    setup_train_label = cv0_lis[1].copy()
    setup_train_batchname = cv0_lis[2].copy()
    setup_train_dismatch_index_pred = cv0_lis[3].copy()
    setup_train_dismatch_index_type = cv0_lis[4].copy()
    setup_train_real_node_label = cv0_lis[5].copy()

    setup_train_df = [cv0_lis[6], cv1_lis[6], cv2_lis[6]]
    setup_train_df = pd.concat(setup_train_df)


    for i in range(0, len(cv1_lis[0])):
      setup_train_data.append(cv1_lis[0][i])
      setup_train_label.append(cv1_lis[1][i])
      setup_train_batchname.append(cv1_lis[2][i])
      setup_train_dismatch_index_pred[list(cv1_lis[3])[i]] = list(cv1_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv1_lis[4])[i]] = list(cv1_lis[4].values())[i]
      setup_train_real_node_label.append(cv1_lis[5][i])


    for i in range(0, len(cv2_lis[0])):
      setup_train_data.append(cv2_lis[0][i])
      setup_train_label.append(cv2_lis[1][i])
      setup_train_batchname.append(cv2_lis[2][i])
      setup_train_dismatch_index_pred[list(cv2_lis[3])[i]] = list(cv2_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv2_lis[4])[i]] = list(cv2_lis[4].values())[i]
      setup_train_real_node_label.append(cv2_lis[5][i])


    # The validation set
    setup_val_data = cv3_lis[0].copy()
    setup_val_label = cv3_lis[1].copy()
    setup_val_batchname = cv3_lis[2].copy()
    setup_val_dismatch_index_pred = cv3_lis[3].copy()
    setup_val_dismatch_index_type = cv3_lis[4].copy()
    setup_val_real_node_label = cv3_lis[5].copy()
    setup_val_df = cv3_lis[6].copy()


    # The test set
    setup_test_data = cv4_lis[0].copy()
    setup_test_label = cv4_lis[1].copy()
    setup_test_batchname = cv4_lis[2].copy()
    setup_test_dismatch_index_pred = cv4_lis[3].copy()
    setup_test_dismatch_index_type = cv4_lis[4].copy()
    setup_test_real_node_label = cv4_lis[5].copy()
    setup_test_df = cv4_lis[6].copy()










  # for setup 2: cv1, cv2, cv3 for train, cv4 for validation, cv0 for test
  elif setup == "setup 2":
    setup_train_data = cv1_lis[0].copy()
    setup_train_label = cv1_lis[1].copy()
    setup_train_batchname = cv1_lis[2].copy()
    setup_train_dismatch_index_pred = cv1_lis[3].copy()
    setup_train_dismatch_index_type = cv1_lis[4].copy()
    setup_train_real_node_label = cv1_lis[5].copy()

    setup_train_df = [cv1_lis[6], cv2_lis[6], cv3_lis[6]]
    setup_train_df = pd.concat(setup_train_df)


    for i in range(0, len(cv2_lis[0])):
      setup_train_data.append(cv2_lis[0][i])
      setup_train_label.append(cv2_lis[1][i])
      setup_train_batchname.append(cv2_lis[2][i])
      setup_train_dismatch_index_pred[list(cv2_lis[3])[i]] = list(cv2_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv2_lis[4])[i]] = list(cv2_lis[4].values())[i]
      setup_train_real_node_label.append(cv2_lis[5][i])


    for i in range(0, len(cv3_lis[0])):
      setup_train_data.append(cv3_lis[0][i])
      setup_train_label.append(cv3_lis[1][i])
      setup_train_batchname.append(cv3_lis[2][i])
      setup_train_dismatch_index_pred[list(cv3_lis[3])[i]] = list(cv3_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv3_lis[4])[i]] = list(cv3_lis[4].values())[i]
      setup_train_real_node_label.append(cv3_lis[5][i])


    # The validation set
    setup_val_data = cv4_lis[0].copy()
    setup_val_label = cv4_lis[1].copy()
    setup_val_batchname = cv4_lis[2].copy()
    setup_val_dismatch_index_pred = cv4_lis[3].copy()
    setup_val_dismatch_index_type = cv4_lis[4].copy()
    setup_val_real_node_label = cv4_lis[5].copy()
    setup_val_df = cv4_lis[6].copy()


    # The test set
    setup_test_data = cv0_lis[0].copy()
    setup_test_label = cv0_lis[1].copy()
    setup_test_batchname = cv0_lis[2].copy()
    setup_test_dismatch_index_pred = cv0_lis[3].copy()
    setup_test_dismatch_index_type = cv0_lis[4].copy()
    setup_test_real_node_label = cv0_lis[5].copy()
    setup_test_df = cv0_lis[6].copy()







  # for setup 3: cv2, cv3, cv4 for train, cv0 for validation, cv1 for test
  elif setup == "setup 3":
    setup_train_data = cv2_lis[0].copy()
    setup_train_label = cv2_lis[1].copy()
    setup_train_batchname = cv2_lis[2].copy()
    setup_train_dismatch_index_pred = cv2_lis[3].copy()
    setup_train_dismatch_index_type = cv2_lis[4].copy()
    setup_train_real_node_label = cv2_lis[5].copy()

    setup_train_df = [cv2_lis[6], cv3_lis[6], cv4_lis[6]]
    setup_train_df = pd.concat(setup_train_df)


    for i in range(0, len(cv3_lis[0])):
      setup_train_data.append(cv3_lis[0][i])
      setup_train_label.append(cv3_lis[1][i])
      setup_train_batchname.append(cv3_lis[2][i])
      setup_train_dismatch_index_pred[list(cv3_lis[3])[i]] = list(cv3_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv3_lis[4])[i]] = list(cv3_lis[4].values())[i]
      setup_train_real_node_label.append(cv3_lis[5][i])


    for i in range(0, len(cv4_lis[0])):
      setup_train_data.append(cv4_lis[0][i])
      setup_train_label.append(cv4_lis[1][i])
      setup_train_batchname.append(cv4_lis[2][i])
      setup_train_dismatch_index_pred[list(cv4_lis[3])[i]] = list(cv4_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv4_lis[4])[i]] = list(cv4_lis[4].values())[i]
      setup_train_real_node_label.append(cv4_lis[5][i])


    # The validation set
    setup_val_data = cv0_lis[0].copy()
    setup_val_label = cv0_lis[1].copy()
    setup_val_batchname = cv0_lis[2].copy()
    setup_val_dismatch_index_pred = cv0_lis[3].copy()
    setup_val_dismatch_index_type = cv0_lis[4].copy()
    setup_val_real_node_label = cv0_lis[5].copy()
    setup_val_df = cv0_lis[6].copy()


    # The test set
    setup_test_data = cv1_lis[0].copy()
    setup_test_label = cv1_lis[1].copy()
    setup_test_batchname = cv1_lis[2].copy()
    setup_test_dismatch_index_pred = cv1_lis[3].copy()
    setup_test_dismatch_index_type = cv1_lis[4].copy()
    setup_test_real_node_label = cv1_lis[5].copy()
    setup_test_df = cv1_lis[6].copy()







  # for setup 4: cv3, cv4, cv0 for train, cv1 for validation, cv2 for test
  elif setup == "setup 4":
    setup_train_data = cv3_lis[0].copy()
    setup_train_label = cv3_lis[1].copy()
    setup_train_batchname = cv3_lis[2].copy()
    setup_train_dismatch_index_pred = cv3_lis[3].copy()
    setup_train_dismatch_index_type = cv3_lis[4].copy()
    setup_train_real_node_label = cv3_lis[5].copy()

    setup_train_df = [cv3_lis[6], cv4_lis[6], cv0_lis[6]]
    setup_train_df = pd.concat(setup_train_df)


    for i in range(0, len(cv4_lis[0])):
      setup_train_data.append(cv4_lis[0][i])
      setup_train_label.append(cv4_lis[1][i])
      setup_train_batchname.append(cv4_lis[2][i])
      setup_train_dismatch_index_pred[list(cv4_lis[3])[i]] = list(cv4_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv4_lis[4])[i]] = list(cv4_lis[4].values())[i]
      setup_train_real_node_label.append(cv4_lis[5][i])


    for i in range(0, len(cv0_lis[0])):
      setup_train_data.append(cv0_lis[0][i])
      setup_train_label.append(cv0_lis[1][i])
      setup_train_batchname.append(cv0_lis[2][i])
      setup_train_dismatch_index_pred[list(cv0_lis[3])[i]] = list(cv0_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv0_lis[4])[i]] = list(cv0_lis[4].values())[i]
      setup_train_real_node_label.append(cv0_lis[5][i])


    # The validation set
    setup_val_data = cv1_lis[0].copy()
    setup_val_label = cv1_lis[1].copy()
    setup_val_batchname = cv1_lis[2].copy()
    setup_val_dismatch_index_pred = cv1_lis[3].copy()
    setup_val_dismatch_index_type = cv1_lis[4].copy()
    setup_val_real_node_label = cv1_lis[5].copy()
    setup_val_df = cv1_lis[6].copy()


    # The test set
    setup_test_data = cv2_lis[0].copy()
    setup_test_label = cv2_lis[1].copy()
    setup_test_batchname = cv2_lis[2].copy()
    setup_test_dismatch_index_pred = cv2_lis[3].copy()
    setup_test_dismatch_index_type = cv2_lis[4].copy()
    setup_test_real_node_label = cv2_lis[5].copy()
    setup_test_df = cv2_lis[6].copy()






  # for setup 5: cv4, cv0, cv1 for train, cv2 for validation, cv3 for test
  elif setup == "setup 5":
    setup_train_data = cv4_lis[0].copy()
    setup_train_label = cv4_lis[1].copy()
    setup_train_batchname = cv4_lis[2].copy()
    setup_train_dismatch_index_pred = cv4_lis[3].copy()
    setup_train_dismatch_index_type = cv4_lis[4].copy()
    setup_train_real_node_label = cv4_lis[5].copy()

    setup_train_df = [cv4_lis[6], cv0_lis[6], cv1_lis[6]]
    setup_train_df = pd.concat(setup_train_df)


    for i in range(0, len(cv0_lis[0])):
      setup_train_data.append(cv0_lis[0][i])
      setup_train_label.append(cv0_lis[1][i])
      setup_train_batchname.append(cv0_lis[2][i])
      setup_train_dismatch_index_pred[list(cv0_lis[3])[i]] = list(cv0_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv0_lis[4])[i]] = list(cv0_lis[4].values())[i]
      setup_train_real_node_label.append(cv0_lis[5][i])


    for i in range(0, len(cv1_lis[0])):
      setup_train_data.append(cv1_lis[0][i])
      setup_train_label.append(cv1_lis[1][i])
      setup_train_batchname.append(cv1_lis[2][i])
      setup_train_dismatch_index_pred[list(cv1_lis[3])[i]] = list(cv1_lis[3].values())[i]
      setup_train_dismatch_index_type[list(cv1_lis[4])[i]] = list(cv1_lis[4].values())[i]
      setup_train_real_node_label.append(cv1_lis[5][i])


    # The validation set
    setup_val_data = cv2_lis[0].copy()
    setup_val_label = cv2_lis[1].copy()
    setup_val_batchname = cv2_lis[2].copy()
    setup_val_dismatch_index_pred = cv2_lis[3].copy()
    setup_val_dismatch_index_type = cv2_lis[4].copy()
    setup_val_real_node_label = cv2_lis[5].copy()
    setup_val_df = cv2_lis[6].copy()


    # The test set
    setup_test_data = cv3_lis[0].copy()
    setup_test_label = cv3_lis[1].copy()
    setup_test_batchname = cv3_lis[2].copy()
    setup_test_dismatch_index_pred = cv3_lis[3].copy()
    setup_test_dismatch_index_type = cv3_lis[4].copy()
    setup_test_real_node_label = cv3_lis[5].copy()
    setup_test_df = cv3_lis[6].copy()



  return setup_train_data, setup_train_label, setup_train_batchname, setup_train_dismatch_index_pred, setup_train_dismatch_index_type, setup_train_real_node_label, setup_train_df, \
  setup_val_data, setup_val_label, setup_val_batchname, setup_val_dismatch_index_pred, setup_val_dismatch_index_type, setup_val_real_node_label, setup_val_df, \
  setup_test_data, setup_test_label, setup_test_batchname, setup_test_dismatch_index_pred, setup_test_dismatch_index_type, setup_test_real_node_label, setup_test_df



# In[ ]:


# Select setup

setup_train_data, setup_train_label, setup_train_batchname, setup_train_dismatch_index_pred, setup_train_dismatch_index_type, setup_train_real_node_label, setup_train_df, \
  setup_val_data, setup_val_label, setup_val_batchname, setup_val_dismatch_index_pred, setup_val_dismatch_index_type, setup_val_real_node_label, setup_val_df,\
  setup_test_data, setup_test_label, setup_test_batchname, setup_test_dismatch_index_pred, setup_test_dismatch_index_type, setup_test_real_node_label, setup_test_df = make_splits(cv0_lis, cv1_lis, cv2_lis, cv3_lis, cv4_lis, "setup 5")


# In[24]:


# Find max len

#max_len = 0
#for i in range(0, len(setup1_train_data)):
#  if len(setup1_train_data[i]['x']) > max_len:
#    max_len = len(setup1_train_data[i]['x'])

#print(max_len)


# In[3]:


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="DL_tmp", #项目名称
    entity="transmembrane-topology", # 用户名
    group="CV setup 5, major voting, schnet", # 对比实验分组
    name= "CV setup 5 with batchsize 1, early stopping and exp learning schedule", #实验的名字
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0003,
    "architecture": "schnet",
    "dataset": "protein 3D structures ",
    "epochs":150,
    'batch_size':1,
    'hidden_channels' :128,
    'weight_decay': 1e-4,
    'max_num_neighbors': 32
    }
)
sns.set_style("whitegrid")


# In[25]:


# # Initialize without pretrained weight
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
# #max_len=max(train_max_len,val_max_len,test_SP_TM_max_len,test_TM_max_len,test_BETA_max_len)+1 #StaticEmbedding need max_len
# # put model to GPU
model = SchNetModel(hidden_channels=128, out_dim=6, max_len=30000, max_num_neighbors=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003,weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1) # Learning schedule added


# implement EarlyStopping: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif validation_loss == self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
                
        return False


# In[ ]:


# Add Gaussian smoothing module
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

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





total_epochs=150
draw_num = 1
global_step = 0

# setup 1
# cv0, cv1, cv2 for train, cv3 for validation, cv4 for test
# the training data is called setup1_train_data
early_stopper = EarlyStopper(patience=5, min_delta=0.001) # se min uprise
epoch_atom_level_accuracy_record_train = []
epoch_loss_record_train=[]
epoch_node_level_accuracy_record_train = []
epoch_atom_level_accuracy_record_val = []
epoch_loss_record_val = []
epoch_node_level_accuracy_record_val = []
epochs = []

smoothing = GaussianSmoothing(6, 29, 5)
for epoch in range(total_epochs):
    epochs.append(epoch)
    epoch_atom_level_accuracy_train = []
    epoch_loss_train=[]
    epoch_node_level_accuracy_train = []
    # train
    for i, data in enumerate(setup_train_data):  
        global_step += 1 
        optimizer.zero_grad()  
        outputs = model(data.to(device))   # put batch data in GPU get logits
        prediction = outputs["node_embedding"]  
        real_label = torch.argmax(torch.tensor(setup_train_label[i]), dim=1).to(device) # put label in GPU
        #loss = criterion(prediction, real_label)  # operate in the same device
        
        predicted = torch.reshape(prediction.to('cpu'), (1,prediction.shape[1], prediction.shape[0]))
        predicted = F.pad(predicted, (14, 14), mode='reflect')
        predicted = smoothing(predicted)
        prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))
        
        loss = criterion(prediction_Gauss.to(device), real_label) # operate in the same device
        
        loss.backward()     
        optimizer.step()    

        #calulate atom-level accuracy and node-level accuracy
        #_, predicted = torch.max(prediction, 1)
        _, predicted = torch.max(prediction_Gauss.to(device), 1)
        correct = (predicted == real_label).sum().item()
        total = real_label.size(0)
        atom_level_accuracy =  correct / total

        # below is operated under CPU node
        processor = MapAtomNode(predicted.cpu(),setup_train_batchname[i],setup_train_dismatch_index_pred,setup_train_dismatch_index_type,setup_train_df)

        train_predict_node_label = processor.map_atom_node() 
        node_level_accuracy = node_accuracy(train_predict_node_label,setup_train_real_node_label[i])

        wandb.log({'train_loss_step':loss.item(), 'global_step':global_step})
        wandb.log({'train_atom_level_accuracy_step':atom_level_accuracy,  'global_step':global_step})
        wandb.log({'train_node_level_accuracy_step':node_level_accuracy, 'global_step':global_step})

        epoch_loss_train.append(loss.item())
        epoch_atom_level_accuracy_train.append(atom_level_accuracy)
        epoch_node_level_accuracy_train.append(node_level_accuracy)
        
    epoch_loss_record_train.append(np.mean(epoch_loss_train))
    epoch_atom_level_accuracy_record_train.append(np.mean(epoch_atom_level_accuracy_train))
    epoch_node_level_accuracy_record_train.append(np.mean(epoch_node_level_accuracy_train))

    wandb.log({'train_loss_epoch':np.mean(epoch_loss_train), 'global_step':global_step})
    wandb.log({'train_atom_level_accuracy_epoch':np.mean(epoch_atom_level_accuracy_train),  'global_step':global_step})
    wandb.log({'train_node_level_accuracy_epoch':np.mean(epoch_node_level_accuracy_train), 'global_step':global_step})
    
    # val
    model.eval()  
    with torch.no_grad():  

        epoch_atom_level_accuracy_val = []
        epoch_loss_val = []
        epoch_node_level_accuracy_val = []
        
        for i, data in enumerate(setup_val_data):  
            outputs = model(data.to(device))
            prediction = outputs["node_embedding"]
            real_label = torch.argmax(torch.tensor(setup_val_label[i]), dim=1).to(device)
            #loss = criterion(prediction, real_label)  # operate in the same device
            
            predicted = torch.reshape(prediction.to('cpu'), (1,prediction.shape[1], prediction.shape[0]))
            predicted = F.pad(predicted, (14, 14), mode='reflect')
            predicted = smoothing(predicted)
            prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))
            
            loss = criterion(prediction_Gauss.to(device), real_label) # operate in the same device
            
            #_, predicted = torch.max(prediction, 1)
            _, predicted = torch.max(prediction_Gauss.to(device), 1)
            correct = (predicted == real_label).sum().item()
            total = real_label.size(0)
            atom_level_accuracy = correct / total

            processor = MapAtomNode(predicted.cpu(), setup_val_batchname[i], setup_val_dismatch_index_pred, setup_val_dismatch_index_type, setup_val_df)
            val_predict_node_label = processor.map_atom_node()
            node_level_accuracy = node_accuracy(val_predict_node_label, setup_val_real_node_label[i])

            epoch_loss_val.append(loss.item())
            epoch_atom_level_accuracy_val.append(atom_level_accuracy)
            epoch_node_level_accuracy_val.append(node_level_accuracy)
            
            wandb.log({'val_loss_step':loss.item(), 'global_step':global_step})
            wandb.log({'val_atom_level_accuracy_step':atom_level_accuracy,  'global_step':global_step})
            wandb.log({'val_node_level_accuracy_step':node_level_accuracy, 'global_step':global_step})


            epoch_loss_val.append(loss.item())
            epoch_atom_level_accuracy_val.append(atom_level_accuracy)
            epoch_node_level_accuracy_val.append(node_level_accuracy)
            
        epoch_loss_record_val.append(np.mean(epoch_loss_val))
        epoch_atom_level_accuracy_record_val.append(np.mean(epoch_atom_level_accuracy_val))
        epoch_node_level_accuracy_record_val.append(np.mean(epoch_node_level_accuracy_val))

        wandb.log({'val_loss_epoch':np.mean(epoch_loss_val), 'global_step':global_step})
        wandb.log({'val_atom_level_accuracy_epoch':np.mean(epoch_atom_level_accuracy_val), 'global_step':global_step})
        wandb.log({'val_node_level_accuracy_epoch':np.mean(epoch_node_level_accuracy_val), 'global_step':global_step})

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if early_stopper.early_stop(np.mean(epoch_loss_val)):             
            break

    if epoch >= 100:
        scheduler.step() # apply learning schedule
    if epoch % draw_num == 0:
        print(f"EPOCH:{epoch}:Train Loss:{np.mean(epoch_loss_train)} Train Atom Level Accuracy:{np.mean(epoch_atom_level_accuracy_train)} Train Node Level Accuracy:{np.mean(epoch_node_level_accuracy_train)}")
        print(f"EPOCH:{epoch}:Val Loss:{np.mean(epoch_loss_val)} Val Atom Level Accuracy:{np.mean(epoch_atom_level_accuracy_val)} Val Node Level Accuracy:{np.mean(epoch_node_level_accuracy_val)}")


wandb.finish()


print("Finished training.")

torch.save(model.state_dict(), '/work3/s194408/Project/result/schnet/CV_setup5_Gauss_result.pth')

   
# Save the validation and training acc
node_acc_results = np.concatenate([ [np.array(epochs)], [np.array(epoch_node_level_accuracy_record_train)], [np.array(epoch_node_level_accuracy_record_val)] ])
np.savetxt("/work3/s194408/Project/result/schnet/CVsetup5_node_acc_results_Gauss.csv", node_acc_results, delimiter=',', comments="", fmt='%s')

loss_results = np.concatenate([ [np.array(epochs)], [np.array(epoch_loss_record_train)], [np.array(epoch_loss_record_val)] ])
np.savetxt("/work3/s194408/Project/result/schnet/CVsetup5_loss_results_Gauss.csv", loss_results, delimiter=',', comments="", fmt='%s')

        



# In[ ]:





# In[ ]:




