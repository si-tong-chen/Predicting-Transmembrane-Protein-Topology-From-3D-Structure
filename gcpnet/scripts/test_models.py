import sys

sys.path.append("/Users/chensitong/DL/gcpnet")
sys.path.append("/Users/chensitong/DL/gcpnet/models")
sys.path.append("/Users/chensitong/DL/gcpnet/scripts")
import pickle
import os
from tasks.predict_topology import CreateDataBeforeBatch
from test import TMPTest


def test():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    path = os.path.join(root_dir, "data", "processed", "dataframe", "cv4.pickle")
    with open(path, "rb") as file:
        cv4 = pickle.load(file)

    TM_name_list = cv4[cv4["protein_type"] == "TM"]["uniprot_id_low"].tolist()
    BETA_name_list = cv4[cv4["protein_type"] == "BETA"]["uniprot_id_low"].tolist()
    SP_TM_name_list = cv4[cv4["protein_type"] == "SP+TM"]["uniprot_id_low"].tolist()

    file_name = "DeepTMHMM.3line"
    path = "/Users/chensitong/DL/gcpnet"
    batch_size = 1
    setup = "setup1"  # choose crossvalidation (total 5)
    modelpath = os.path.join(root_dir, "models", "final_model", "CVsetup1_model_major_voting_size1_epoch50.pth")

    processsor = CreateDataBeforeBatch(path)
    _, _, test_data_dict_before_batch = processsor.get_data(setup)

    TM_test = {}
    for name in TM_name_list:
        TM_test[name] = test_data_dict_before_batch[name]

    BETA_test = {}
    for name in BETA_name_list:
        BETA_test[name] = test_data_dict_before_batch[name]

    SP_TM_test = {}
    for name in SP_TM_name_list:
        SP_TM_test[name] = test_data_dict_before_batch[name]
    ##TM
    processor = TMPTest(TM_test, file_name, path, batch_size, 5, setup="setup1", modelpath=modelpath)
    processor.printresult()

    # SP_TM
    processor = TMPTest(SP_TM_test, file_name, path, batch_size, 5, setup="setup1", modelpath=modelpath)
    processor.printresult()

    ##BETA
    processor = TMPTest(BETA_test, file_name, path, batch_size, 3, setup="setup1", modelpath=modelpath)
    processor.printresult()

    # all the test data
    processor = TMPTest(
        test_data_dict_before_batch, file_name, path, batch_size, 3, setup="setup1", modelpath=modelpath
    )
    processor.printresult()


if __name__ == "__main__":
    test()
