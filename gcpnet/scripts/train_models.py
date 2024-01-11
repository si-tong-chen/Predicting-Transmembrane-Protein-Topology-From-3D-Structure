import sys

sys.path.append("/Users/chensitong/DL/gcpnet")
sys.path.append("/Users/chensitong/DL/gcpnet/models")
from models.gcpnet import GCPNetModel
import numpy as np
import os
import torch
import hydra
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tasks.predict_topology import (
    LableTogther,
    MapAtomNode,
    node_accuracy,
    ProcessBatch,
    GaussianSmoothing,
)
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main.yaml", version_base="1.3.2")
def _main(cfg: DictConfig):
    enc = hydra.utils.instantiate(cfg)
    model = GCPNetModel(**enc.models)
    # train(enc.train,enc.labels,model)


def train(cfg_t, cfg, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg_t.lr, weight_decay=cfg_t.weight_decay)

    processor = LableTogther(cfg)
    (
        train_residual_level_label,
        train_atom_levl_label,
        train_dismatch_index_pred,
        train_dismatch_index_type,
        val_residual_level_label,
        val_atom_levl_label,
        val_dismatch_index_pred,
        val_dismatch_index_type,
        df_train,
        df_val,
    ) = processor.processinglabel2()
    train_data_loader, val_data_loader, _ = processor.processinglabel()

    epoch_atom_level_accuracy_record_train = []
    epoch_loss_record_train = []
    epoch_residual_level_accuracy_record_train = []
    epoch_atom_level_accuracy_record_val = []
    epoch_loss_record_val = []
    epoch_residual_level_accuracy_record_val = []

    smoothing = GaussianSmoothing(6, 29, 5)

    for epoch in range(cfg_t.epochs):
        epoch_atom_level_accuracy_train = []
        epoch_loss_train = []
        epoch_residual_level_accuracy_train = []
        # train
        for data_batch in train_data_loader:
            global_step += 1
            batchname = [data_batch[num]["name"] for num in range(len(data_batch))]
            label_part = [value.unsqueeze(0) for name in batchname for value in train_atom_levl_label[name].to_dense()]
            atom_levl_label = torch.cat(label_part).to(device)
            residual_level_label = [value for name in batchname for value in train_residual_level_label[name]]

            batchprocessor = ProcessBatch()
            data = batchprocessor.batchdata(data_batch)
            optimizer.zero_grad()
            outputs = model(data.to(device))
            prediction = outputs["node_embedding"]

            predicted = torch.reshape(prediction.to("cpu"), (1, prediction.shape[1], prediction.shape[0]))
            predicted = F.pad(predicted, (14, 14), mode="reflect")
            predicted = smoothing(predicted)
            prediction_Gauss = torch.reshape(predicted, (prediction.shape[0], prediction.shape[1]))

            loss = criterion(prediction_Gauss.to(device), atom_levl_label)
            loss.backward()
            optimizer.step()

            # calulate atom-level accuracy and node-level accuracy
            _, predicted = torch.max(prediction_Gauss.to(device), 1)
            correct = (predicted == atom_levl_label).sum().item()
            total = atom_levl_label.size(0)
            atom_level_accuracy = correct / total

            processor = MapAtomNode(
                predicted.cpu(), batchname, train_dismatch_index_pred, train_dismatch_index_type, df_train
            )
            train_predict_node_label = processor.map_atom_node()
            residual_level_accuracy = node_accuracy(train_predict_node_label, residual_level_label)

            epoch_loss_train.append(loss.item())
            epoch_atom_level_accuracy_train.append(atom_level_accuracy)
            epoch_residual_level_accuracy_train.append(residual_level_accuracy)

        epoch_loss_record_train.append(np.mean(epoch_loss_train))
        epoch_atom_level_accuracy_record_train.append(np.mean(epoch_atom_level_accuracy_train))
        epoch_residual_level_accuracy_record_train.append(np.mean(epoch_residual_level_accuracy_train))

        # val
        model.eval()
        with torch.no_grad():
            epoch_atom_level_accuracy_val = []
            epoch_loss_val = []
            epoch_residual_level_accuracy_val = []

            for data_batch in val_data_loader:
                batchname = [data_batch[num]["name"] for num in range(len(data_batch))]
                label_part = [
                    value.unsqueeze(0) for name in batchname for value in val_atom_levl_label[name].to_dense()
                ]
                atom_levl_label = torch.cat(label_part).to(device)
                residual_level_label = [value for name in batchname for value in val_residual_level_label[name]]
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

                processor = MapAtomNode(
                    predicted.cpu(), batchname, val_dismatch_index_pred, val_dismatch_index_type, df_val
                )
                val_predict_node_label = processor.map_atom_node()
                residual_level_accuracy = node_accuracy(val_predict_node_label, residual_level_label)

                epoch_loss_val.append(loss.item())
                epoch_atom_level_accuracy_val.append(atom_level_accuracy)
                epoch_residual_level_accuracy_val.append(residual_level_accuracy)

            epoch_loss_record_val.append(np.mean(epoch_loss_val))
            epoch_atom_level_accuracy_record_val.append(np.mean(epoch_atom_level_accuracy_val))
            epoch_residual_level_accuracy_record_val.append(np.mean(epoch_residual_level_accuracy_val))

    print("Finished training.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    modelpath = os.path.join(root_dir, "models", "final_model", "CVsetup1_model_major_voting_size1_epoch50.pth")

    torch.save(model.state_dict(), modelpath)
    print("epoch_residual_level_accuracy_record_train", epoch_residual_level_accuracy_record_train)
    print("epoch_residual_level_accuracy_record_val", epoch_residual_level_accuracy_record_val)
    print("epoch_loss_record_train", epoch_loss_record_train)
    print("epoch_loss_record_val", epoch_loss_record_val)


if __name__ == "__main__":
    _main()
