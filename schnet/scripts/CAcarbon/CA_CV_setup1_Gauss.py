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
train_data,train_lable, train_max_len,train_real_node_label,train_CA_index_list,train_atoms_length = processor.datalabelgenerator('train')

val_data,val_lable,val_max_len,val_real_node_label,val_CA_index_list,val_atoms_length = processor.datalabelgenerator('val')

test_SP_TM_data,test_SP_TM_lable, test_SP_TM_max_len,test_SP_TM_real_node_label,test_SP_TM_CA_index_list,test_SP_TM_atoms_length = processor.datalabelgenerator('test_SP_TM')

test_TM_data,test_TM_lable,test_TM_max_len,test_TM_real_node_label,test_TM_CA_index_list,test_TM_atoms_length = processor.datalabelgenerator('test_TM')

test_BETA_data,test_BETA_lable,test_BETA_max_len,test_BETA_real_node_label,test_BETA_CA_index_list,test_BETA_atoms_length = processor.datalabelgenerator('test_BETA')


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


# In[12]:


import pandas as pd

# train_data,train_lable, train_max_len,train_real_node_label,train_CA_index_list,train_atoms_length

# Group the data together
total_data = train_data.copy()
total_label = train_lable.copy()
#total_batchname = train_batchname.copy()
total_max_len = train_max_len + val_max_len + test_SP_TM_max_len + test_TM_max_len + test_BETA_max_len
#total_dismatch_index_pred = train_dismatch_index_pred.copy()
#total_dismatch_index_type = train_dismatch_index_type.copy()
total_real_node_label = train_real_node_label.copy()
total_CA_index_list = train_CA_index_list.copy()
total_atoms_length = train_atoms_length.copy()

#frames = [df_train, df_val, df_test_SP_TM, df_test_TM, df_test_BETA]
#total_df = pd.concat(frames)

#train_data,train_lable, train_batchname, train_max_len,train_dismatch_index_pred,train_dismatch_index_type,train_real_node_label,df_train 
#test_SP_TM_data,test_SP_TM_lable, test_SP_TM_batchname, test_SP_TM_max_len,test_SP_TM_dismatch_index_pred,test_SP_TM_dismatch_index_type,test_SP_TM_real_node_label,test_SP_TM_val
#test_TM_data,test_TM_lable, test_TM_batchname, test_TM_max_len,test_TM_dismatch_index_pred,test_TM_dismatch_index_type,test_TM_real_node_label,test_TM_val 
#test_BETA_data,test_BETA_lable, test_BETA_batchname, test_BETA_max_len,test_BETA_dismatch_index_pred,test_BETA_dismatch_index_type,test_BETA_real_node_label,test_BETA_val
for i in range(0, len(val_data)):
    total_data.append(val_data[i])
    total_label.append(val_lable[i])
    #total_batchname.append(val_batchname[i])
    #total_dismatch_index_pred[list(val_dismatch_index_pred)[i]] = list(val_dismatch_index_pred.values())[i]
    #total_dismatch_index_type[list(val_dismatch_index_type)[i]] = list(val_dismatch_index_type.values())[i]
    total_real_node_label.append(val_real_node_label[i])
    total_CA_index_list.append(val_CA_index_list[i])
    total_atoms_length.append(val_atoms_length[i])


for i in range(0, len(test_SP_TM_data)):
    total_data.append(test_SP_TM_data[i])
    total_label.append(test_SP_TM_lable[i])
#     total_batchname.append(test_SP_TM_batchname[i])
#     total_dismatch_index_pred[list(test_SP_TM_dismatch_index_pred)[i]] = list(test_SP_TM_dismatch_index_pred.values())[i]
#     total_dismatch_index_type[list(test_SP_TM_dismatch_index_type)[i]] = list(test_SP_TM_dismatch_index_type.values())[i]
    total_real_node_label.append(test_SP_TM_real_node_label[i])
    total_CA_index_list.append(test_SP_TM_CA_index_list[i])
    total_atoms_length.append(test_SP_TM_atoms_length[i])


for i in range(0, len(test_TM_data)):
    total_data.append(test_TM_data[i])
    total_label.append(test_TM_lable[i])
  #  total_batchname.append(test_TM_batchname[i])
   # total_dismatch_index_pred[list(test_TM_dismatch_index_pred)[i]] = list(test_TM_dismatch_index_pred.values())[i]
   # total_dismatch_index_type[list(test_TM_dismatch_index_type)[i]] = list(test_TM_dismatch_index_type.values())[i]
    total_real_node_label.append(test_TM_real_node_label[i])
    total_CA_index_list.append(test_TM_CA_index_list[i])
    total_atoms_length.append(test_TM_atoms_length[i])


for i in range(0, len(test_BETA_data)):
    total_data.append(test_BETA_data[i])
    total_label.append(test_BETA_lable[i])
    #total_batchname.append(test_BETA_batchname[i])
    #total_dismatch_index_pred[list(test_BETA_dismatch_index_pred)[i]] = list(test_BETA_dismatch_index_pred.values())[i]
    #total_dismatch_index_type[list(test_BETA_dismatch_index_type)[i]] = list(test_BETA_dismatch_index_type.values())[i]
    total_real_node_label.append(test_BETA_real_node_label[i])
    total_CA_index_list.append(test_BETA_CA_index_list[i])
    total_atoms_length.append(test_BETA_atoms_length[i])



# In[14]:


total_x_len = []
for i in range(0, len(total_data)):
    total_x_len.append(len(total_data[i].x))


# In[19]:


cv0_x_len = [3438,1849,1320,3000,2248,1396,5312,2606,2217,2116,4573,7038,1067,1932,3063,8961,15111,2480,2692,2903,2534,2617,4651,2063,9064,2476,1466,2168,9531,2767,960,2806,2428,1929,2177,3010,1050,4500,2918,2983,3711,2839,487,1206,557,3555,2192,2840,1298,4585,2861,4371,2606,921,3336,2450,3658,2074,2573,3555,618,4418,3400,1420,1561,2163,3503,2396,1476,1422,3279,3743,2494,5301,829,2125,3488,909,2845,3906,1066,1229,4777,2547,3770,3147,3787,1482,4007,2509,2647,913,8753,1970,1608,2221,5078,1737,2327,1904,957,5572,761,3361,3943,2238,744,810,1301,9983,14943,4595,1851,5273,4222,4169,6315,6260,1929,4592,4783,1025,8498,1073,1200,1084,6552,4608,8444,655,4293,3959,4223,2862,3204,3508,1519,4137,392,6387,4406,9370,7132,2102,2809,3059,9481,2370,5818,4020,2515,9488,898,2141,7631,4172,3911,1245,6553,10695,4719,995,3909,976,3812,3176,8723,2797,1701,2837,1643,1481,3825,1215,8109,1525,1954,1800,3731,3812,4229,2902,835,4912,9807,2323,2167,1012,2570,4247,3379,563,940,2647,2920,5325,5862,2826,3392,11493,1708,1129,4401,2458,5014,1224,2086,3998,871,4546,5085,5240,1030,757,730,3264,1422,3817,2349,977,6874,2710,17757,4050,1165,5902,4041,2545,3287,2872,6621,5593,4651,12445,2516,3270,3930,493,4665,1084,1587,7674,8412,2796,4610,2027,2396,3145,1959,1114,2361,3407,2195,3561,1485,7993,14873,1527,3600,2219,2675,8679,7149,4497,2747,863,8942,3214,1036,931,2969,1103,2019,10127,4509,3728,2408,1196,8491,2248,12241,1450,6958,2940,6914,5913,3531,3663,5967,3595,1395,5610,1088,3211,5613,3197,8504,1131,5959,3126,2975,465,2902,2614,1400,6355,1985,1092,13146,1094,2076,6405,1794,4610,5833,1439,1691,616,4443,2762,8482,913,8543,4134,9340,6826,2434,1850,4192,4206,3438,3198,10525,5865,3039,678,3268,4438,2649,9776,1406,4287,6088,4182,3982,2078,2066,707,5720,6076,5947,14041,8480,9587,1789,1332,5117,5389,2097,6655,5658,12333,1345,1250,7167,2237,2415,5516,3723,3198,3198,10795,2861,18350,876,5895,18917,4020,5829,2929,2971,4440,2457,602,2988,1191,3205,3409,2710,3319,1119,1329,765,2243,1903,2141,14661,3918,6736,14969,1110,3112,2508,3512,2576,14686,762,1925,3047,1018,1557,4117,4559,1900,2500,2122,3169,1364,1334,12048,2801,1902,2220,5514,4538,977,2276,4676,628,2963,1903,1190,7660,2406,2379,406,6715,4880,7882,3512,2110,2467,4781,1894,6896,2219,4840,3077,2669,4674,713,1130,2976,2564,3652,9085,2786,2635,6503,2935,2969,10899,5853,561,11668,455,6361,6352,3095,1628,1519,893,5500,3216,1740,3423,1201,1751,16761,469,3266,495,5149,3550,17574,2606,5681,2594,1249,4471,3237,4046,1799,1042,2130,3995,4044,2572,1526,1455,10601,5176,2506,3798,6216,2429,2135,771,1647,1371,2565,4731,3010,2723,2496,3508,8625,1448,3434,6072,2246,8380,3228,913,2422,1448,3730,1871,1244,4966,2660,807,4685,3768,649,1860,2671,1348,9555,4255,4314,1103,5090,4969,3516,518,591,2514,3739,1153,4143,3225,2244,3174,1014,3090,4811,2395,2970,4635,4136,2612,2236,1994,848,3569,9522,9723,2091,8731,7675,4277,5009,4569,2579,3940,7013,525,4892,3639,1360,3185,3575,2799,1039,1793,1676,4838,3643,1199,2239,4212,1854,3916,987,4614,1483,3108,2746,3183,3310,2732,4669,4093,1634,2563,12127,5653,4991,6645,2840,11402,3907,1902,2665,1539,13438,3371,2330,2362,678,870,2402,7178,5533,3923,3025,7676,3950,3833,1449,1348,3375,3399,4933,3182,2435,4852,664,10520,4300,2884,2648,5292,10775,1521,2708,3710,2690,6379,3814,3114,3692,7278,688,2272,3141,2783,3435,2595,1677,6668,6673,2706,2979,2995,6414,3480,2447,6632,3287,1283,4421,6763,3577,1368,4875,2688,8564,2447,12348,3898,1551,5367,6838,6152,17374,4123,3551,15916,2724,886,4159,933,4611,5623,2699,1434,1962,3384,3008,3479,3140,1076,2058,4630,8182,1499,4910,2521,2065,5739,1491]
cv1_x_len = [3257,509,12059,9938,3356,3554,2217,1250,1211,2971,2517,807,6052,5169,1202,2144,3883,624,4765,5077,2530,5865,2790,6697,5438,806,1178,989,2386,673,6625,4890,5493,3163,6927,2923,10939,2521,2743,7158,8446,1702,8858,1429,4107,1221,2132,1780,2286,5455,7944,7254,3157,18763,2922,2861,816,4916,6875,1308,3468,4119,1782,2739,1064,3895,3334,7822,1312,6670,1807,2946,1259,4970,6429,3315,1252,6048,3783,4413,932,4634,1998,3317,2854,2681,5837,9560,2704,4377,2994,7464,4396,1428,2730,13428,10318,2862,2258,4784,7480,14934,3603,6415,4695,1809,1519,3180,4722,6263,4378,4137,5906,9669,3098,3109,1659,2960,2902,2104,849,5510,6289,1978,1176,1158,1192,2769,3999,6592,4442,4537,3142,3030,7257,472,4055,5911,2567,4903,3192,4383,9524,2472,5384,5474,1002,2972,8865,9192,5593,470,3343,3009,3040,1074,2771,1891,3050,4344,2624,5023,1782,4612,1819,3592,2060,2688,1336,5593,4842,1538,704,6440,1835,1059,5532,1253,1383,3143,1117,3366,1817,3713,1007,1827,3002,4856,6635,2214,2885,1381,844,2566,1269,17691,3322,6023,6242,1150,7572,826,7338,2224,2548,2563,2944,5599,774,4173,695,8976,1702,3976,1989,3386,1399,8403,3298,5822,4156,1875,5517,4883,1990,3176,5575,7443,3907,2266,2059,1886,1712,1596,5206,6445,896,3286,2337,3625,3312,992,1001,766,3081,7886,7573,6580,852,4437,2208,1876,1317,4963,2271,12071,5075,3352,3446,995,849,2851,3109,1847,8708,2360,1238,1410,2910,6646,2413,1477,2222,1652,13485,2666,2262,3915,3650,6146,2199,865,1427,539,3464,542,2201,842,1422,1993,7937,5347,3458,7558,3864,851,3129,2934,2016,3156,840,5851,413,4861,2907,4179,1646,3846,3766,849,7704,5679,4280,1820,1701,1933,1195,1391,10712,5929,2623,3361,2176,4370,2404,1925,3199,822,2407,6028,6902,758,7357,1653,3683,2612,3000,2561,1887,1854,2489,2527,4021,7483,1090,1314,4199,4106,6679,879,4278,3252,703,4039,2750,3358,3519,3119,3770,1022,13384,756,1633,3817,2639,3269,8471,5403,1975,2379,1641,2068,6428,3293,2833,1025,8133,11376,860,8812,1992,2489,3540,2277,11245,653,5737,1965,1253,484,881,2518,2820,6783,12768,1659,4426,7366,3313,1903,675,5107,7596,3935,4292,1110,9797,14148,2154,7866,781,3590,2065,2130,11762,3427,6164,1297,4705,1103,2507,1653,3144,1468,4008,5165,2704,985,3790,2560,1065,4086,2017,3193,5007,3677,3096,13159,5454,894,942,4762,1143,2404,2801,7347,1891,542,5774,904,2875,4153,3906,4725,753,675,807,6346,1913,8349,2236,1289,2504,3682,3655,8077,2956,1705,18900,1982,2010,4230,722,11157,10107,7347,605,1945,1396,8052,1960,3368,705,2761,4341,7311,3082,2687,2046,2299,4304,4204,4947,1914,2006,5573,776,4472,7333,2791,1924,3945,3818,6034,3762,1708,3161,2893,1047,1354,5223,5942,7285,898,10594,1631,2115,2838,18370,2204,3536,3537,2796,2667,3726,6457,3372,2077,1118,4180,1213,2754,607,2855,2823,2036,869,1110,464,5345,3907,633,9535,5707,3201,858,1930,1973,643,5319,2410,12868,2030,2136,7301,8231,2149,3253,1040,2230,1917,9470,517,3007,2373,1611,978,3456,2135,7178,2540,1404,3189,4677,2852,6265,2015,1353,5374,5365,5065,9032,3881,4805,2869,1035,1063,1201,1772,3295,1178,4420,1109,4329,2187,2268,1037,1325,3759,2190,1824,2389,1451,5101,13696,1681,3609,17331,1756,1664,9930,1851,13935,477,676,2829,3815,3447,767,2011,3326,651,3783,1939,2692,2074,2122,1211,3607,6823,3234,2427,834,2100,1344,5136,4443,7388,2079,3567,2353,4304,8108,1627,2591,4511,7094,3391,2966,4731,2999,5104,2998,3149,2306,2379,3565,3789,4174,7919,14455,2489,2935,5047,2915,4434,7410,4056,9329,3168,6809,2214,1496,1201,2818,6347,1610,5723,980,768,2365,2615,1112,769,2510,2650,5426,3433,1464,3353,1304,1920,1502,6190,630,4191,3653,12814,11000,3961,1135,4013,3799,4114,5352,6032,8619,4377,3292,3972]
cv2_x_len = [2683,9466,1745,2488,1793,4060,2319,3133,3565,3670,1642,4847,7794,8495,1576,3373,2293,2771,3181,2023,3164,1194,1659,4374,479,2306,5326,7984,3540,2611,5086,4642,3175,5273,11128,823,1870,4526,10457,3259,4885,9120,1355,2906,5851,937,2255,2892,4073,748,10628,1286,1198,4467,3661,6242,3080,1320,4863,2716,3135,1226,4019,2471,3295,1778,417,3498,7574,869,2749,5017,2784,2402,2523,6408,808,5876,1683,1624,5627,577,6591,4220,5142,7387,6830,1523,11814,1884,3771,3925,18030,1573,2369,4867,1154,3055,2289,2929,5420,1771,6908,1356,3134,5185,530,1733,3491,2316,4460,2429,9270,640,2737,1825,6645,1214,5268,2255,5229,1434,4008,3078,3777,2995,4160,5463,3644,7610,2994,4190,4147,20566,5195,1722,4417,7337,1071,8616,10326,762,9808,1223,11718,2224,4579,12585,5137,3401,4538,1055,3016,6509,6458,13241,8905,3697,6365,3053,555,1581,3257,5733,10029,1822,7134,3153,1539,2635,6934,1861,8231,3446,1630,932,3353,6155,7896,2606,1754,4834,4535,3996,4136,9826,5860,1705,1304,571,5135,1200,4663,3459,10969,4586,1439,1859,2457,1072,4688,4869,2793,5211,5623,2833,9074,2215,6924,10678,1141,6536,1920,1811,6020,1597,807,3533,1004,4584,3247,2189,5382,2700,3587,3585,1855,9839,6519,2576,7462,2227,2519,1926,10770,3073,15066,3118,5181,1256,3663,699,2072,3629,1886,629,982,1680,6805,3409,3749,8402,4044,2480,3966,2664,3128,1914,4436,1987,5811,2786,3735,3283,2588,3705,2737,5551,2272,9311,2021,2403,9128,2609,1221,2175,4417,2727,1551,587,5811,1226,2870,2206,8670,2897,2581,1314,782,8356,9960,1742,2757,903,5478,12347,6548,2083,4547,787,2041,1888,3682,8086,3122,8516,5256,5924,1685,5823,4602,2048,806,3634,419,2074,4216,3728,1780,2496,2303,10040,4268,1363,1185,12807,4738,4777,2035,2660,712,2510,2236,1282,2236,2523,4141,4668,1764,3141,7873,6830,2675,8288,995,2361,11772,927,2682,4964,4874,1385,8004,4006,3097,9824,1356,5795,1715,5158,10982,4085,15135,3356,934,2570,7748,3835,1554,4527,1342,1774,2591,7574,2339,2193,4578,1765,1917,2618,3305,794,6352,3014,3401,5069,7828,3141,2612,1470,1954,3813,657,3153,1677,1083,1809,1689,2786,933,1405,4128,4283,2926,1403,913,3652,2128,1022,2634,12964,4699,6119,7601,984,1832,2154,2630,5786,2786,6705,1823,2072,670,9225,2721,1731,5476,1345,9826,5260,5409,3152,5353,4690,1589,5398,2398,2238,2571,7489,1679,1859,1938,5145,825,4225,1002,8976,4144,575,5328,1477,2284,4761,2863,4214,2452,2388,1373,4222,2274,5120,2371,1296,2947,1586,6215,1001,4003,6148,2449,4056,3596,3387,1451,1639,2871,4384,1477,5446,5176,2983,3989,2143,3271,5844,1062,1919,2069,3349,2866,1611,2567,5203,10657,3382,1669,7167,3550,1617,7302,3311,842,2789,1095,13538,962,2416,2754,3914,3610,3331,1969,3181,8106,3668,1786,1110,2211,1606,6504,2836,3271,1436,2200,3267,3816,797,3496,610,2348,1962,1206,1008,2980,833,641,13667,3826,967,2782,6315,3992,1595,9533,1953,2518,4026,5538,1845,3733,3270,2014,7117,487,600,2773,1614,13312,4544,7196,847,2070,3206,573,2983,1045,595,1358,5797,10011,5637,4427,9282,9658,3982,4028,2247,2231,7080,3006,1511,15343,2675,1346,1924,3436,4746,2376,896,2353,20789,2773,7138,1274,3082,2260,1773,5025,6055,5404,1526,3968,2090,2909,3782,2222,489,1920,6873,769,5018,3009,2473,4049,2000,1795,4255,1219,1550,662,2102,2832,2150,10023,3075,4673,3087,3177,4531,3133,3510,2444,3852,778,5635,5341,1899,4968,6780,3832,3142,3844,3052,2615,7237,2577,12633,3199,5486,2180,2083,544,1653,6245,2704,3764,1966,2780,1029,1784,2623,3164,504,3995,1915,4752,11682,8131,2484,6709,1836,1959,5714,2682,5523,544,2009,1953,6039,794,19972,904,7647,1009,4578,3037,7310,5315,5902,4582,5323,1660,2683,2972,6137,6191,2194,723,3861,2669,5723,4458]
cv3_x_len = [696,1517,9845,1627,712,2098,4673,2161,679,12286,8651,10444,2365,7718,1163,2704,4195,7110,6487,2768,3100,1907,4202,1312,780,1655,2115,3689,1548,3893,1410,490,4153,715,2171,11187,3782,5503,1441,3484,6265,2345,1127,2238,2264,1887,5786,16868,2993,3561,8210,3885,1664,10886,733,4329,7824,6513,6389,1687,2303,1183,7502,1404,1420,6395,1223,2856,5002,1327,1887,3690,1495,12732,5523,1820,2747,2819,1242,2866,884,4449,9914,2724,1735,3195,5869,1283,5422,5911,859,4657,5909,4826,2280,3578,3065,1295,2877,3200,2188,2879,2817,4592,7886,1704,11067,819,7081,1629,7591,7141,2862,611,3684,5577,2486,6813,3780,2782,512,4803,6391,1484,1018,8314,4177,1803,1776,4289,992,6985,1588,1775,9565,4407,1832,2703,2808,9080,431,2586,4410,1905,5726,3538,755,3238,1370,3883,1967,8570,2499,1725,12456,1862,1740,1473,7782,606,1377,11838,2179,2193,863,4325,489,7564,4094,4645,3516,2795,4550,843,2989,1547,1199,2352,5273,5317,6809,4322,1066,4886,3043,2911,7158,3666,2905,8780,2400,2516,2560,1421,2869,3212,4901,1081,2003,3787,1645,2044,893,4410,4034,3557,3138,1874,3996,4351,4839,840,2231,2331,6103,6022,1155,5371,5537,1504,5443,8118,2922,2164,3583,8439,3611,3328,1906,4214,922,1622,2692,9081,730,4270,2755,1702,1697,2563,5576,1700,3012,500,2246,6831,6238,4369,3172,5598,6959,2369,9655,917,7957,1512,1595,3989,3276,7125,2055,2692,3792,3098,4708,6742,1511,7422,930,13542,5368,1945,9333,2005,1285,7319,1151,512,3279,1776,2090,2839,12856,4466,3340,480,1225,1320,1513,3126,2596,4098,3818,3762,2837,4482,2577,2401,9219,1172,12131,8689,2820,2822,4412,3874,8055,3236,4748,1055,626,842,3604,3849,2409,5786,9415,6147,4157,866,6359,2738,1332,1743,7128,7204,8164,2820,1821,431,3869,3832,1995,1173,5298,3678,672,2672,9044,3842,2577,10271,4814,4046,2562,3387,1091,8144,4481,6086,1430,4288,1097,3849,11316,1189,2476,6664,3506,4098,18382,6772,2253,3369,916,7002,4600,4203,1792,847,3771,791,12052,3972,2021,1592,1543,2380,3716,1631,2627,12979,984,9000,8397,2870,1425,4827,10783,4368,1062,5971,8026,3408,4181,2106,2255,767,5495,3230,634,3619,6656,520,4074,2611,5301,4947,2763,1524,16712,5089,5945,3837,6936,13292,2025,723,2261,1518,2596,2765,3299,4664,3340,19164,3914,4042,1442,5834,1335,2530,6425,3054,3228,3292,1617,5686,7314,6065,1189,604,2694,11408,4682,8925,1683,1328,1525,3596,12656,10960,5850,4445,3277,14934,1907,2306,11768,1692,1880,3196,1103,4572,6271,1437,5419,4724,981,1736,1115,1889,2299,11544,3405,1831,1674,3141,3830,1672,2280,2658,9339,2152,601,961,2330,2811,7263,3098,4528,1239,2739,4297,4325,1154,1542,2459,1134,990,1927,2765,3719,7227,9284,1289,2241,2317,4513,5444,1765,2408,5190,3457,3476,1340,2410,2960,2363,1525,4220,1661,5442,10250,1390,3823,1820,2075,9087,4760,16152,6533,3973,1616,686,5668,3094,9807,2709,1884,1448,1517,17245,1637,5147,2242,1320,4629,3634,4452,2227,2987,595,4108,905,9114,1190,2405,1750,3596,15649,2991,1415,3942,8277,4036,2410,1513,4443,5524,3291,2648,3230,2198,4145,2232,1279,5252,2074,400,4072,529,3943,2323,10305,1014,11707,2912,1499,1311,2628,2021,3365,4519,7992,1448,1839,5850,3328,4377,2579,4041,3881,5439,5886,607,678,3917,3968,2133,1976,2534,2656,2856,2668,4190,17936,4679,5797,7554,2838,1800,1704,1962,2996,1575,1681,4897,1482,4838,1535,4057,17949,1653,7445,3257,3763,1313,7621,12090,2789,4385,1712,13934,833,4030,7204,3016,7141,1179,6988,1771,3749,990,1914,1140,4299,1693,2261,2771,1685,1457,2257,2540,5185,3413,672,2480,4161,3554,4749,4109,9648,1919,1569,4590,2488,8583,5714,4281,5160,2616,5165,3110,3250,16704,3735,9696,1505,5523,3839,2058,1392,6471,831,3492,5602,13077,8062,1951,6330,2403,3715,1597]
cv4_x_len = [2494,6795,1923,9592,734,3095,7511,661,2244,541,5225,5436,3085,1571,12622,2538,2208,4141,5452,4517,5774,1024,2580,3954,7662,3691,18657,2784,2164,519,1178,9916,1863,3000,2844,1208,1158,2614,8170,1768,3107,2235,616,5282,4558,5587,3674,3876,2253,7633,3477,1697,6997,4895,4592,1070,2438,20118,3192,1258,5817,1002,4016,3502,2544,10460,12521,4328,3511,2757,5800,3478,1987,7948,2718,1623,3776,6537,1688,5429,1445,12598,1362,4058,992,7411,2278,3292,2555,1527,447,2088,575,3755,561,3419,1900,5685,3299,3296,4108,478,11256,1343,5036,1138,1252,2517,2381,1380,589,2958,5525,5474,4015,2987,2268,2353,5468,1791,5244,1360,11746,2888,2009,3856,2656,4338,12816,1982,2494,7767,2398,1227,885,10191,4128,3758,3795,3354,5264,1838,3262,7508,9367,1564,1578,11871,657,1182,1257,3723,6359,2586,5014,4071,5191,2586,6776,3359,1721,4012,2300,1985,2233,5766,2598,5930,7645,1139,3141,2703,2335,421,505,1101,2792,1082,3659,4468,1401,2035,1589,2036,2899,2083,1101,1866,2586,955,541,2967,14330,4645,2289,2934,3222,3160,4571,6750,3564,1976,6759,2348,1270,2778,3959,2990,9180,1830,4406,3269,1036,4811,2794,5605,5287,6342,1462,792,6184,1104,2462,3305,11146,1187,7398,1211,2804,1275,4653,5830,3417,477,6361,4362,4529,8506,2552,3200,3711,5693,5765,3356,666,6806,4265,4341,3924,761,6533,3716,3411,2053,1708,19974,8826,1176,9057,11052,2870,970,1227,3552,8352,2625,2252,2406,4274,6045,1436,2336,2574,3907,5316,3917,4683,2157,4419,8652,2325,3903,3794,2899,8947,2834,7630,3752,734,3801,3353,1508,1151,2558,3195,10166,3714,13250,4657,5458,5251,3543,1914,4676,5581,2529,1095,12343,7937,2824,2217,3584,1707,1643,642,6348,668,3682,4674,1679,6155,7282,1314,6040,5381,8522,1415,3100,8318,2537,2662,9668,3544,2415,6838,1023,12726,702,13035,3385,1789,4485,2863,7279,5589,980,1734,2862,845,2423,3008,2155,4410,1537,4813,618,5630,11676,9135,12002,2240,9245,3338,2323,2920,2236,2393,2316,1165,1499,3706,4862,4891,7112,1871,4592,3055,531,5879,1787,6672,2926,7669,1233,6862,3322,1913,3343,4466,468,13394,3127,947,8845,5249,489,1341,5003,5529,5366,1427,14764,6302,3580,3167,1475,6802,3727,2785,8477,438,6923,1851,8022,3766,4762,1103,5216,5370,5126,1508,4292,6131,8284,3322,4633,736,744,1400,9822,550,6297,2601,11530,1521,3978,11711,6218,3148,644,926,2965,4461,17411,2488,2850,6261,4000,3493,2697,2997,1579,5021,2572,2531,1509,1211,1136,4939,2072,1954,1461,10098,6453,2828,5686,3013,12045,4994,3349,2460,2879,976,1564,2020,1528,2863,3163,5514,4412,4606,648,5635,2400,1731,5127,4552,1431,3295,3194,859,7414,1981,1313,3749,926,2180,1879,1379,960,7840,5213,1009,3577,2076,11313,2450,4149,5335,3485,2809,945,1817,2213,2325,427,3902,3554,1799,1031,6770,2325,754,3940,530,3346,1314,2778,3432,3505,2275,4534,5193,1120,3300,7318,781,791,8172,522,2959,3456,545,4972,14391,1314,2725,2703,6550,1942,1738,1746,3624,3259,4234,4779,6506,4515,4037,2518,2633,4514,10122,5026,15138,1450,3127,1936,4445,6211,1184,1002,4112,1527,1301,1414,2675,1419,1110,1488,6557,2567,9229,2270,1299,1072,11604,3545,15023,1296,3192,3598,7413,2044,1703,1237,3116,2558,1491,3953,3912,1620,1382,1563,6998,1212,6607,2085,7385,2187,7967,2148,4498,2281,5593,3375,4225,3463,2976,730,609,1478,16372,3219,992,3334,7216,2150,1761,2661,3416,3353,3406,600,4214,5005,2284,2049,2735,1053,4339,1361,1756,4806,2084,3032,3640,1708,2143,10806,1560,4392,2032,6150,1892,10596,1547,4051,1074,4619,4148,4624,3624,1136,4878,1913,9062,2078,1778,2566,5258,11659,712,2481,1081,3796,2880,2773,1478,1128,2208,3969,2512,1967,3565,1327,3020,1570,3518,3407,490,2883,8966,8113,4591,20223,2789,3818,4010,1265,2204,3005,783,1769,2104,4248,1128,6076,6728,420,2032]


# In[24]:


# For cv0
cv0_data = []
cv0_label = []
# cv0_batchname = []
# cv0_dismatch_index_pred = {}
# cv0_dismatch_index_type = {}
cv0_real_node_label = []
cv0_index = []
#cv0_df = pd.DataFrame(columns=total_df.columns)
cv0_CA_index_list = []
cv0_total_atoms_length = []



# Find index for the found cv0 proteins
# Note that the labels are aligned with the batch names
for i in range(0, len(cv0_x_len)):
    cv0_index.append(total_x_len.index(cv0_x_len[i]))




# gather the data for cv0
for i in range(0, len(cv0_index)):
    cv0_data.append(total_data[cv0_index[i]])
    cv0_label.append(total_label[cv0_index[i]])
    #cv0_dismatch_index_pred[list(total_dismatch_index_pred)[cv0_index[i]]] = list(total_dismatch_index_pred.values())[cv0_index[i]]
    #cv0_dismatch_index_type[list(total_dismatch_index_type)[cv0_index[i]]] = list(total_dismatch_index_type.values())[cv0_index[i]]
    cv0_real_node_label.append(total_real_node_label[cv0_index[i]])
    cv0_CA_index_list.append(total_CA_index_list[cv0_index[i]])
    cv0_total_atoms_length.append(total_atoms_length[cv0_index[i]])


# put into a list
cv0_lis = [cv0_data, cv0_label, cv0_real_node_label, cv0_CA_index_list, cv0_total_atoms_length]




# for cv1
cv1_data = []
cv1_label = []
cv1_real_node_label = []
cv1_index = []
cv1_CA_index_list = []
cv1_total_atoms_length = []



# Find index for the found cv0 proteins
for i in range(0, len(cv1_x_len)):
    cv1_index.append(total_x_len.index(cv1_x_len[i]))




# gather the data for cv0
for i in range(0, len(cv1_index)):
    cv1_data.append(total_data[cv1_index[i]])
    cv1_label.append(total_label[cv1_index[i]])
    cv1_real_node_label.append(total_real_node_label[cv1_index[i]])
    cv1_CA_index_list.append(total_CA_index_list[cv1_index[i]])
    cv1_total_atoms_length.append(total_atoms_length[cv1_index[i]])

# put into a list
cv1_lis = [cv1_data, cv1_label, cv1_real_node_label, cv1_CA_index_list, cv1_total_atoms_length]
    
    

# for cv2
cv2_data = []
cv2_label = []
cv2_real_node_label = []
cv2_index = []
cv2_CA_index_list = []
cv2_total_atoms_length = []



# Find index for the found cv0 proteins
for i in range(0, len(cv2_x_len)):
    cv2_index.append(total_x_len.index(cv2_x_len[i]))




# gather the data for cv0
for i in range(0, len(cv2_index)):
    cv2_data.append(total_data[cv2_index[i]])
    cv2_label.append(total_label[cv2_index[i]])
    cv2_real_node_label.append(total_real_node_label[cv2_index[i]])
    cv2_CA_index_list.append(total_CA_index_list[cv2_index[i]])
    cv2_total_atoms_length.append(total_atoms_length[cv2_index[i]])


# put into a list
cv2_lis = [cv2_data, cv2_label, cv2_real_node_label, cv2_CA_index_list, cv2_total_atoms_length]

    
    
    
    
# for cv3
cv3_data = []
cv3_label = []
cv3_real_node_label = []
cv3_index = []
cv3_CA_index_list = []
cv3_total_atoms_length = []



# Find index for the found cv0 proteins
for i in range(0, len(cv3_x_len)):
    cv3_index.append(total_x_len.index(cv3_x_len[i]))




# gather the data for cv0
for i in range(0, len(cv3_index)):
    cv3_data.append(total_data[cv3_index[i]])
    cv3_label.append(total_label[cv3_index[i]])
    cv3_real_node_label.append(total_real_node_label[cv3_index[i]])
    cv3_CA_index_list.append(total_CA_index_list[cv3_index[i]])
    cv3_total_atoms_length.append(total_atoms_length[cv3_index[i]])


# put into a list
cv3_lis = [cv3_data, cv3_label, cv3_real_node_label, cv3_CA_index_list, cv3_total_atoms_length]



    
    
    
    
# for cv4
cv4_data = []
cv4_label = []
cv4_real_node_label = []
cv4_index = []
cv4_CA_index_list = []
cv4_total_atoms_length = []



# Find index for the found cv0 proteins
for i in range(0, len(cv4_x_len)):
    cv4_index.append(total_x_len.index(cv4_x_len[i]))




# gather the data for cv0
for i in range(0, len(cv4_index)):
    cv4_data.append(total_data[cv4_index[i]])
    cv4_label.append(total_label[cv4_index[i]])
    cv4_real_node_label.append(total_real_node_label[cv4_index[i]])
    cv4_CA_index_list.append(total_CA_index_list[cv4_index[i]])
    cv4_total_atoms_length.append(total_atoms_length[cv4_index[i]])
    

# put into a list
cv4_lis = [cv4_data, cv4_label, cv4_real_node_label, cv4_CA_index_list, cv4_total_atoms_length]

    
#print(len(cv0_data), len(cv1_data), len(cv2_data), len(cv3_data), len(cv4_data))


# In[55]:


def make_splits(cv0_lis, cv1_lis, cv2_lis, cv3_lis, cv4_lis, setup):

    # for setup 1:
    #cv0, cv1, cv2 for train, cv3 for validation, cv4 for test
    if setup == "setup 1":
    # data format
    #cv0_lis = [cv0_data, cv0_label, cv0_batchname, cv0_dismatch_index_pred, cv0_dismatch_index_type, cv0_real_node_label, cv0_df]

        setup_train_data = cv0_lis[0].copy()
        setup_train_label = cv0_lis[1].copy()
        setup_train_real_node_label = cv0_lis[2].copy()
        setup_train_CA_index_list = cv0_lis[3].copy()
        setup_train_total_atoms_length = cv0_lis[4].copy()




        for i in range(0, len(cv1_lis[0])):
            setup_train_data.append(cv1_lis[0][i])
            setup_train_label.append(cv1_lis[1][i])
            setup_train_real_node_label.append(cv1_lis[2][i])
            setup_train_CA_index_list.append(cv1_lis[3][i])
            setup_train_total_atoms_length.append(cv1_lis[4][i])


        for i in range(0, len(cv2_lis[0])):
            setup_train_data.append(cv2_lis[0][i])
            setup_train_label.append(cv2_lis[1][i])
            setup_train_real_node_label.append(cv2_lis[2][i])
            setup_train_CA_index_list.append(cv2_lis[3][i])
            setup_train_total_atoms_length.append(cv2_lis[4][i])


        # The validation set
        setup_val_data = cv3_lis[0].copy()
        setup_val_label = cv3_lis[1].copy()
        setup_val_real_node_label = cv3_lis[2].copy()
        setup_val_CA_index_list = cv3_lis[3].copy()
        setup_val_total_atoms_length = cv3_lis[4].copy()


        # The test set
        setup_test_data = cv4_lis[0].copy()
        setup_test_label = cv4_lis[1].copy()
        setup_test_real_node_label = cv4_lis[2].copy()
        setup_test_CA_index_list = cv4_lis[3].copy()
        setup_test_total_atoms_length = cv4_lis[4].copy()










    # for setup 2: cv1, cv2, cv3 for train, cv4 for validation, cv0 for test
    elif setup == "setup 2":

        setup_train_data = cv1_lis[0].copy()
        setup_train_label = cv1_lis[1].copy()
        setup_train_real_node_label = cv1_lis[2].copy()
        setup_train_CA_index_list = cv1_lis[3].copy()
        setup_train_total_atoms_length = cv1_lis[4].copy()



        for i in range(0, len(cv2_lis[0])):
            setup_train_data.append(cv2_lis[0][i])
            setup_train_label.append(cv2_lis[1][i])
            setup_train_real_node_label.append(cv2_lis[2][i])
            setup_train_CA_index_list.append(cv2_lis[3][i])
            setup_train_total_atoms_length.append(cv2_lis[4][i])


        for i in range(0, len(cv3_lis[0])):
            setup_train_data.append(cv3_lis[0][i])
            setup_train_label.append(cv3_lis[1][i])
            setup_train_real_node_label.append(cv3_lis[2][i])
            setup_train_CA_index_list.append(cv3_lis[3][i])
            setup_train_total_atoms_length.append(cv3_lis[4][i])


        # The validation set
        setup_val_data = cv4_lis[0].copy()
        setup_val_label = cv4_lis[1].copy()
        setup_val_real_node_label = cv4_lis[2].copy()
        setup_val_CA_index_list = cv4_lis[3].copy()
        setup_val_total_atoms_length = cv4_lis[4].copy()


        # The test set
        setup_test_data = cv0_lis[0].copy()
        setup_test_label = cv0_lis[1].copy()
        setup_test_real_node_label = cv0_lis[2].copy()
        setup_test_CA_index_list = cv0_lis[3].copy()
        setup_test_total_atoms_length = cv0_lis[4].copy()







    # for setup 3: cv2, cv3, cv4 for train, cv0 for validation, cv1 for test
    elif setup == "setup 3":

        setup_train_data = cv2_lis[0].copy()
        setup_train_label = cv2_lis[1].copy()
        setup_train_real_node_label = cv2_lis[2].copy()
        setup_train_CA_index_list = cv2_lis[3].copy()
        setup_train_total_atoms_length = cv2_lis[4].copy()




        for i in range(0, len(cv3_lis[0])):
            setup_train_data.append(cv3_lis[0][i])
            setup_train_label.append(cv3_lis[1][i])
            setup_train_real_node_label.append(cv3_lis[2][i])
            setup_train_CA_index_list.append(cv3_lis[3][i])
            setup_train_total_atoms_length.append(cv3_lis[4][i])


        for i in range(0, len(cv4_lis[0])):
            setup_train_data.append(cv4_lis[0][i])
            setup_train_label.append(cv4_lis[1][i])
            setup_train_real_node_label.append(cv4_lis[2][i])
            setup_train_CA_index_list.append(cv4_lis[3][i])
            setup_train_total_atoms_length.append(cv4_lis[4][i])


        # The validation set
        setup_val_data = cv0_lis[0].copy()
        setup_val_label = cv0_lis[1].copy()
        setup_val_real_node_label = cv0_lis[2].copy()
        setup_val_CA_index_list = cv0_lis[3].copy()
        setup_val_total_atoms_length = cv0_lis[4].copy()


        # The test set
        setup_test_data = cv1_lis[0].copy()
        setup_test_label = cv1_lis[1].copy()
        setup_test_real_node_label = cv1_lis[2].copy()
        setup_test_CA_index_list = cv1_lis[3].copy()
        setup_test_total_atoms_length = cv1_lis[4].copy()







    # for setup 4: cv3, cv4, cv0 for train, cv1 for validation, cv2 for test
    elif setup == "setup 4":
        setup_train_data = cv3_lis[0].copy()
        setup_train_label = cv3_lis[1].copy()
        setup_train_real_node_label = cv3_lis[2].copy()
        setup_train_CA_index_list = cv3_lis[3].copy()
        setup_train_total_atoms_length = cv3_lis[4].copy()




        for i in range(0, len(cv4_lis[0])):
            setup_train_data.append(cv4_lis[0][i])
            setup_train_label.append(cv4_lis[1][i])
            setup_train_real_node_label.append(cv4_lis[2][i])
            setup_train_CA_index_list.append(cv4_lis[3][i])
            setup_train_total_atoms_length.append(cv4_lis[4][i])


        for i in range(0, len(cv0_lis[0])):
            setup_train_data.append(cv0_lis[0][i])
            setup_train_label.append(cv0_lis[1][i])
            setup_train_real_node_label.append(cv0_lis[2][i])
            setup_train_CA_index_list.append(cv0_lis[3][i])
            setup_train_total_atoms_length.append(cv0_lis[4][i])


        # The validation set
        setup_val_data = cv1_lis[0].copy()
        setup_val_label = cv1_lis[1].copy()
        setup_val_real_node_label = cv1_lis[5].copy()
        setup_val_CA_index_list = cv1_CA_index_list.copy()
        setup_val_total_atoms_length = cv1_total_atoms_length.copy()


        # The test set
        setup_test_data = cv2_lis[0].copy()
        setup_test_label = cv2_lis[1].copy()
        setup_test_real_node_label = cv2_lis[2].copy()
        setup_test_CA_index_list = cv2_lis[3].copy()
        setup_test_total_atoms_length = cv2_lis[4].copy()






    # for setup 5: cv4, cv0, cv1 for train, cv2 for validation, cv3 for test
    elif setup == "setup 5":
        setup_train_data = cv4_lis[0].copy()
        setup_train_label = cv4_lis[1].copy()
        setup_train_real_node_label = cv4_lis[2].copy()
        setup_train_CA_index_list = cv4_lis[3].copy()
        setup_train_total_atoms_length = cv4_lis[4].copy()




        for i in range(0, len(cv0_lis[0])):
            setup_train_data.append(cv0_lis[0][i])
            setup_train_label.append(cv0_lis[1][i])
            setup_train_real_node_label.append(cv0_lis[2][i])
            setup_train_CA_index_list.append(cv0_lis[3][i])
            setup_train_total_atoms_length.append(cv0_lis[4][i])


        for i in range(0, len(cv1_lis[0])):
            setup_train_data.append(cv1_lis[0][i])
            setup_train_label.append(cv1_lis[1][i])
            setup_train_real_node_label.append(cv1_lis[2][i])
            setup_train_CA_index_list.append(cv1_lis[3][i])
            setup_train_total_atoms_length.append(cv1_lis[4][i])


        # The validation set
        setup_val_data = cv2_lis[0].copy()
        setup_val_label = cv2_lis[1].copy()
        setup_val_real_node_label = cv2_lis[5].copy()
        setup_val_CA_index_list = cv2_CA_index_list.copy()
        setup_val_total_atoms_length = cv2_total_atoms_length.copy()


        # The test set
        setup_test_data = cv3_lis[0].copy()
        setup_test_label = cv3_lis[1].copy()
        setup_test_real_node_label = cv3_lis[2].copy()
        setup_test_CA_index_list = cv3_lis[3].copy()
        setup_test_total_atoms_length = cv3_lis[4].copy()



    return setup_train_data, setup_train_label, setup_train_real_node_label, setup_train_CA_index_list, setup_train_total_atoms_length, \
setup_val_data, setup_val_label, setup_val_real_node_label, setup_val_CA_index_list, setup_val_total_atoms_length, \
setup_test_data, setup_test_label, setup_test_real_node_label, setup_test_CA_index_list, setup_test_total_atoms_length



# In[37]:


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="DL_tmp", #项目名称
    entity="transmembrane-topology", # 用户名
    group="CA carbon, CV setup 1", # 对比实验分组
    name= "batchsize=1 ", #实验的名字
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "schnet",
    "dataset": "protein 3D structures ",
    "epochs":300,
    'batch_size':1,
    'hidden_channels' :128,
    'weight_decay': 1e-4
    }
)
sns.set_style("whitegrid")


# In[62]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len=max(train_max_len,val_max_len,test_SP_TM_max_len,test_TM_max_len,test_BETA_max_len)+1 #StaticEmbedding need max_len
# put model to GPU
model = SchNetModel(hidden_channels=128, out_dim=6, max_len=30000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2) # Learning schedule added


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
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# In[39]:


# Try a torch implementation of Gaussian smoothing
# from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
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
    def __init__(self, channels, kernel_size, sigma, dim=2):
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


# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = F.pad(input, (2, 2, 2, 2), mode='reflect')
# output = smoothing(input)


#smoothing = GaussianSmoothing(1, 7, 1)
# predicted = torch.reshape(prediction.to('cpu'), (1, 1, prediction.shape[1], prediction.shape[0]))
# predicted = F.pad(predicted, (3, 3, 3, 3), mode='reflect')
# predicted = smoothing(predicted)
# output = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))


# >>> t4d = torch.empty(3, 3, 4, 2)
# >>> p1d = (1, 1) # pad last dim by 1 on each side
# >>> out = F.pad(t4d, p1d, "reflect", 0)
#print(input.shape)


# In[56]:


setup_train_data, setup_train_label, setup_train_real_node_label, setup_train_CA_index_list, setup_train_total_atoms_length, \
setup_val_data, setup_val_label, setup_val_real_node_label, setup_val_CA_index_list, setup_val_total_atoms_length, \
setup_test_data, setup_test_label, setup_test_real_node_label, setup_test_CA_index_list, setup_test_total_atoms_length = make_splits(cv0_lis, cv1_lis, cv2_lis, cv3_lis, cv4_lis, "setup 1")


# In[61]:


total_epochs=300
draw_num = 1
global_step = 0



early_stopper = EarlyStopper(patience=5, min_delta=0.001) # se min uprise
epoch_atom_level_accuracy_record_train = []
epoch_loss_record_train=[]
epoch_node_level_accuracy_record_train = []
epoch_atom_level_accuracy_record_val = []
epoch_loss_record_val = []
epoch_node_level_accuracy_record_val = []
epochs = []

smoothing = GaussianSmoothing(6, 29, 5, 1)
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
        
        predicted = torch.reshape(prediction.to('cpu'), (1,prediction.shape[1], prediction.shape[0]))
        predicted = F.pad(predicted, (14, 14), mode='reflect')
        predicted = smoothing(predicted)
        prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))

        loss = criterion(prediction_Gauss.to(device), real_label)
        
        #loss = criterion(prediction, real_label)  # operate in the same device
        loss.backward()     
        optimizer.step()    

        #calulate atom-level accuracy and 
        _, predicted = torch.max(prediction_Gauss.to(device), 1) 
        correct = (predicted == real_label).sum().item()
        total = real_label.size(0)
        atom_level_accuracy =  correct / total

        #node-level accuracy
        j=0
        CA_pred_all=[]
        for k in range(len(setup_train_total_atoms_length[i])):
            index_last = int(setup_train_total_atoms_length[i][k]) + int(j)
            part_pred = predicted[j:index_last]
            CA_pred = [part_pred[index] for index in setup_train_CA_index_list[i][k]]
            CA_pred_all.extend(CA_pred)
            j = setup_train_total_atoms_length[i][k]

        tensor_label = torch.tensor(setup_train_real_node_label[i], dtype=torch.float32).to(device)
        CA_pred_all= [t.unsqueeze(0) for t in CA_pred_all]
        CA_pred_all = torch.cat(CA_pred_all, dim=0)
        node_correct = (CA_pred_all == tensor_label).sum().item()
        node_total = CA_pred_all.size(0)
        node_level_accuracy =  node_correct / node_total

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
            
            predicted = torch.reshape(prediction.to('cpu'), (1,prediction.shape[1], prediction.shape[0]))
            predicted = F.pad(predicted, (14, 14), mode='reflect')
            predicted = smoothing(predicted)
            prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))
            
            loss = criterion(prediction_Gauss.to(device), real_label)
            
            _, predicted = torch.max(prediction_Gauss.to(device), 1)
            correct = (predicted == real_label).sum().item()
            total = real_label.size(0)
            atom_level_accuracy = correct / total

            #node-level accuracy
            j=0
            CA_pred_all=[]
            for k in range(len(setup_val_total_atoms_length[i])):
                index_last = int(setup_val_total_atoms_length[i][k]) + int(j)
                part_pred = predicted[j:index_last]
                CA_pred = [part_pred[index] for index in setup_val_CA_index_list[i][k]]
                CA_pred_all.extend(CA_pred)
                j = setup_val_total_atoms_length[i][k]

            tensor_label = torch.tensor(setup_val_real_node_label[i], dtype=torch.float32).to(device)
            CA_pred_all= [t.unsqueeze(0) for t in CA_pred_all]
            CA_pred_all = torch.cat(CA_pred_all, dim=0)
            node_correct = (CA_pred_all == tensor_label).sum().item()
            node_total = CA_pred_all.size(0)
            node_level_accuracy =  node_correct / node_total
            
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
            
    if epoch >= 46:
        scheduler.step() # apply learning schedule
        
        
    if epoch % draw_num == 0:
        print(f"EPOCH:{epoch}:Train Loss:{np.mean(epoch_loss_train)} Train Atom Level Accuracy:{np.mean(epoch_atom_level_accuracy_train)} Train Node Level Accuracy:{np.mean(epoch_node_level_accuracy_train)}")
        print(f"EPOCH:{epoch}:Val Loss:{np.mean(epoch_loss_val)} Val Atom Level Accuracy:{np.mean(epoch_atom_level_accuracy_val)} Val Node Level Accuracy:{np.mean(epoch_node_level_accuracy_val)}")


wandb.finish()


print("Finished training.")

torch.save(model.state_dict(), '/work3/s194408/Project/result/schnet/CVsetup1_CA_size1.pth')


# Save the validation and training acc
node_acc_results = np.concatenate([ [np.array(epochs)], [np.array(epoch_node_level_accuracy_record_train)], [np.array(epoch_node_level_accuracy_record_val)] ])
np.savetxt("/work3/s194408/Project/result/schnet/CA_CVsetup1_node_acc_results_Gauss.csv", node_acc_results, delimiter=',', comments="", fmt='%s')

loss_results = np.concatenate([ [np.array(epochs)], [np.array(epoch_loss_record_train)], [np.array(epoch_loss_record_val)] ])
np.savetxt("/work3/s194408/Project/result/schnet/CA_CVsetup1_loss_results_Gauss.csv", loss_results, delimiter=',', comments="", fmt='%s')




# In[ ]:





# In[ ]:




