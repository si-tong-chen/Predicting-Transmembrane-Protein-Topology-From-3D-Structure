from tasks.predict_topology import TMPDataset, CreateLable, MapAtomNode, node_accuracy, ProcessBatch, GaussianSmoothing
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from models.gcpnet import GCPNetModel
import torch.optim as optim
from omegaconf import OmegaConf
import numpy as np
from tasks.predict_topology import MapAtomNode, node_accuracy
from scipy.ndimage import gaussian_filter1d
from metrics_utils import label_list_to_topology, is_topologies_equal

# model paramters
num_layers = 4
emb_dim = 64
node_s_emb_dim = emb_dim
node_v_emb_dim = 8
edge_s_emb_dim = 16
edge_v_emb_dim = 2
r_max = 10.0
num_rbf = 8
activation = "silu"
pool = "sum"

module_cfg = OmegaConf.create(
    {
        "norm_pos_diff": True,
        "scalar_gate": 0,
        "vector_gate": True,
        "scalar_nonlinearity": activation,
        "vector_nonlinearity": activation,
        "nonlinearities": [activation, activation],
        "r_max": r_max,
        "num_rbf": num_rbf,
        "bottleneck": 2,
        "vector_linear": True,
        "vector_identity": True,
        "default_bottleneck": 2,
        "predict_node_positions": True,
        "predict_node_rep": True,
        "node_positions_weight": 1.0,
        "update_positions_with_vector_sum": False,
        "enable_e3_equivariance": False,
        "pool": pool,
    }
)

# model_cfg
model_cfg = OmegaConf.create(
    {
        "h_input_dim": 1,
        "chi_input_dim": 2,
        "e_input_dim": 9,
        "xi_input_dim": 1,
        "h_hidden_dim": node_s_emb_dim,
        "chi_hidden_dim": node_v_emb_dim,
        "e_hidden_dim": edge_s_emb_dim,
        "xi_hidden_dim": edge_v_emb_dim,
        "num_layers": num_layers,
        "dropout": 0.0,
    }
)

# layer_cfg
layer_cfg = OmegaConf.create(
    {
        "pre_norm": False,
        "use_gcp_norm": True,
        "use_gcp_dropout": True,
        "use_scalar_message_attention": True,
        "num_feedforward_layers": 2,
        "dropout": 0.0,
        "nonlinearity_slope": 1e-2,
        "mp_cfg": {
            "edge_encoder": False,
            "edge_gate": False,
            "num_message_layers": 4,
            "message_residual": 0,
            "message_ff_multiplier": 1,
            "self_message": True,
        },
    }
)


class TMPTest:
    def __init__(self, TM_test, file_name, path, batch_size, num, setup, modelpath):
        self.TM_test = TM_test
        self.file_name = file_name
        self.path = path
        self.batch_size = batch_size
        self.num = num
        self.setup = setup
        self.modelpath = modelpath

    def testtogther(self):
        test_dataset = TMPDataset(self.TM_test)
        test_data_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=True, collate_fn=lambda x: x, pin_memory=True
        )

        # put test label togther
        test_residual_level_label = {}
        test_atom_levl_label = {}
        test_dismatch_index_pred = {}
        test_dismatch_index_type = {}
        for data_batch in test_data_loader:
            batchname = [data_batch[num]["name"] for num in range(len(data_batch))]

            labelprocessor = CreateLable(batchname, data_batch, self.path, self.file_name)
            (
                atom_level_label_dict,
                redidual_level_label_dict,
                dismatch_index_pred,
                dismatch_index_type,
                _,
                _,
                df_test,
            ) = labelprocessor.labeldispatcher(self.setup, subset="test")

            test_atom_levl_label.update(atom_level_label_dict)
            test_residual_level_label.update(redidual_level_label_dict)
            test_dismatch_index_pred.update(dismatch_index_pred)
            test_dismatch_index_type.update(dismatch_index_type)

        return (
            test_atom_levl_label,
            test_residual_level_label,
            test_dismatch_index_pred,
            test_dismatch_index_type,
            df_test,
        )

    def testmodel(
        self,
    ):
        (
            test_atom_levl_label,
            test_residual_level_label,
            test_dismatch_index_pred,
            test_dismatch_index_type,
            df_test,
        ) = self.testtogther()

        test_dataset = TMPDataset(self.TM_test)
        test_data_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: x, pin_memory=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GCPNetModel(
            num_layers=num_layers,
            node_s_emb_dim=node_s_emb_dim,
            node_v_emb_dim=node_v_emb_dim,
            edge_s_emb_dim=edge_s_emb_dim,
            edge_v_emb_dim=edge_v_emb_dim,
            r_max=r_max,
            num_rbf=num_rbf,
            activation=activation,
            pool=pool,
            module_cfg=module_cfg,
            model_cfg=model_cfg,
            layer_cfg=layer_cfg,
        ).to(device)

        model.load_state_dict(torch.load(self.modelpath))

        criterion = nn.CrossEntropyLoss()
        smoothing = GaussianSmoothing(6, 29, 5)

        test_predict_node_label_lis = []

        check_zero = []
        test_node_acc_list = []
        test_node_acc_binary_list = []

        test_atom_acc_list = []
        test_atom_correct = []
        test_atom_total = []

        baseline_atom_correct = []
        baseline_atom_acc_list = []
        baseline_node_acc_list = []
        setup_test_real_node_label = []
        model.eval()
        with torch.no_grad():
            for data_batch in test_data_loader:
                batchname = [data_batch[num]["name"] for num in range(len(data_batch))]
                label_part = [
                    value.unsqueeze(0) for name in batchname for value in test_atom_levl_label[name].to_dense()
                ]
                atom_levl_label = torch.cat(label_part).to(device)
                residual_level_label = [value for name in batchname for value in test_residual_level_label[name]]
                batchprocessor = ProcessBatch()
                data = batchprocessor.batchdata(data_batch)

                outputs = model(data.to(device))
                prediction = outputs["node_embedding"]

                predicted = torch.reshape(prediction.to("cpu"), (1, prediction.shape[1], prediction.shape[0]))
                predicted = F.pad(predicted, (14, 14), mode="reflect")
                predicted = smoothing(predicted)
                prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))

                loss = criterion(prediction_Gauss.to(device), atom_levl_label)

                _, predicted = torch.max(prediction_Gauss.to(device), 1)
                correct = (predicted == atom_levl_label).sum().item()
                total = atom_levl_label.size(0)
                atom_level_accuracy = correct / total

                test_atom_acc_list.append(atom_level_accuracy)
                test_atom_correct.append(correct)
                test_atom_total.append(total)

                # Baseline atom accuracy
                baseline_atom = torch.zeros_like(
                    atom_levl_label
                )  # note that in this case the most frequent class is class 0
                baseline_correct = (baseline_atom == atom_levl_label).sum().item()
                baseline_atom_correct.append(baseline_correct)

                processor = MapAtomNode(
                    predicted.cpu(), batchname, test_dismatch_index_pred, test_dismatch_index_type, df_test
                )
                test_predict_node_label = processor.map_atom_node()
                test_predict_node_label_lis.append(test_predict_node_label)

                accuracy_list = [1 if x == y else 0 for x, y in zip(test_predict_node_label, residual_level_label)]
                test_node_acc_binary_list += accuracy_list
                residual_level_accuracy = node_accuracy(test_predict_node_label, residual_level_label)
                test_node_acc_list.append(residual_level_accuracy)

                # Baseline node accuracy
                baseline_node_label = np.zeros_like(
                    np.array(residual_level_label)
                )  # note that in this case the most frequent class is class 0
                baseline_accuracy_list = [1 if x == y else 0 for x, y in zip(baseline_node_label, residual_level_label)]
                baseline_node_acc_list += baseline_accuracy_list

                setup_test_real_node_label.append(residual_level_label)

        return (
            test_node_acc_list,
            test_node_acc_list,
            test_node_acc_binary_list,
            test_node_acc_binary_list,
            test_atom_acc_list,
            test_atom_correct,
            test_atom_total,
            test_predict_node_label_lis,
            setup_test_real_node_label,
        )

    def printresult(self):
        (
            test_node_acc_list,
            test_node_acc_list,
            test_node_acc_binary_list,
            test_node_acc_binary_list,
            test_atom_acc_list,
            test_atom_correct,
            test_atom_total,
            test_predict_node_label_lis,
            setup_test_real_node_label,
        ) = self.testmodel()
        final_node_acc = sum(test_node_acc_list) / len(test_node_acc_list)

        print("Node acc:", final_node_acc)

        final_node_binary_acc = sum(test_node_acc_binary_list) / len(test_node_acc_binary_list)
        print("Node binary acc:", final_node_binary_acc)

        final_atom_acc = sum(test_atom_acc_list) / len(test_atom_acc_list)
        print("Avg atom acc:", final_atom_acc)

        total_atom_acc = sum(test_atom_correct) / len(test_atom_total)
        print("Total atom acc:", total_atom_acc)

        test_resul = []
        for i in range(0, len(test_predict_node_label_lis)):
            # for j in range(0, len(test_BETA_predict_node_label_lis[i])):
            topo_A = label_list_to_topology(test_predict_node_label_lis[i])
            topo_B = label_list_to_topology(setup_test_real_node_label[i])

            test_resul.append(is_topologies_equal(topo_A, topo_B, self.num))

        print("Correct topology without Gaussian smoothing:", sum(test_resul) / len(test_resul))

        resul = []
        acc_lis = []
        for i in range(0, len(test_predict_node_label_lis)):
            # for j in range(0, len(test_BETA_predict_node_label_lis[i])):
            topo_A = label_list_to_topology(test_predict_node_label_lis[i])
            topo_B = label_list_to_topology(setup_test_real_node_label[i])

            # print(topo_A)
            # print(topo_B)

            resul.append(is_topologies_equal(topo_A, topo_B, self.num))
            # print(SP_TM_resul.append(is_topologies_equal(topo_A, topo_B, 5)))
            node_level_accuracy = node_accuracy(test_predict_node_label_lis[i], setup_test_real_node_label[i])
            acc_lis.append(node_level_accuracy)

        final_node_acc = sum(acc_lis) / len(acc_lis)
        print("Node acc before smoothing:", final_node_acc)
        print("Before smoothing correct topology:", sum(resul) / len(resul), "\n")

        resul_2 = []
        acc_lis = []
        for i in range(0, len(test_predict_node_label_lis)):
            smoothened = gaussian_filter1d(
                test_predict_node_label_lis[i], sigma=3
            )  # using same Gaussian filter setup as TMBED article

            topo_A = label_list_to_topology(smoothened.tolist())
            topo_B = label_list_to_topology(setup_test_real_node_label[i])

            resul_2.append(is_topologies_equal(topo_A, topo_B, self.num))
            node_level_accuracy = node_accuracy(smoothened.tolist(), setup_test_real_node_label[i])
            acc_lis.append(node_level_accuracy)

        final_node_acc = sum(acc_lis) / len(acc_lis)
        print("Node acc after Gaussian smoothing:", final_node_acc)
        print("After Gaussian smoothing correct topology:", sum(resul_2) / len(resul_2), "\n")
