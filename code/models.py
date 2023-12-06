import pickle

import pandas as pd
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import torchvision.transforms.functional
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import json

from sklearn.neighbors import KNeighborsClassifier

from .config import CLASSES_CNT
from .data_management import DatasetGTSRB, BatchSampler, SamplerForKNN
from .data_management import TestData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#####################
# Models and training
#####################


class SimpleClassifier(pl.LightningModule):
    """
    :param features_criterion: loss for embeddings
    :param internal_features: internal features number
    """

    def __init__(self, features_criterion=None,
                 internal_features=1024,
                 loss=torch.nn.functional.nll_loss):
        super(SimpleClassifier, self).__init__()

        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=internal_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=internal_features, out_features=CLASSES_CNT),
        )

        for child in list(self.model.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False

        self.loss = loss

    def training_step(self, batch, idx):
        x, path, y = batch
        logits = self.model(x)
        pred = torch.nn.functional.log_softmax(logits, dim=1)
        loss = self.loss(pred, y)

        # add accuracy logging
        accuracy = Accuracy(task="multiclass", num_classes=CLASSES_CNT).to(DEVICE)
        self.log_dict({"acc": accuracy(pred, y)},
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True
                      )

        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return [optim]

    def predict(self, x):
        return torch.argmax(self.model(x), dim=1).numpy()


def train_simple_classifier(train_pathes, output_path, classes_json="classes.json", epochs=2):
    """function for training simple classifier on source Train data"""
    dataset = DatasetGTSRB(
        train_pathes,
        classes_json
    )
    dl_train = DataLoader(dataset, batch_size=256, shuffle=True)

    # create and train the model
    model = SimpleClassifier()
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[],
        enable_checkpointing=False,
        logger=False,
        accelerator="auto"
    )
    trainer.fit(model, dl_train)

    # save
    torch.save(model.state_dict(), output_path)

    return model


def apply_classifier(model, test_folder, path_to_classes_json, annotations_file=None):
    """
    function for obtaining model predictions.
    :param model
    :param test_folder: path to folder with data
    :param path_to_classes_json: path to classes.json
    :param annotations_file: path to annotations_file
    """
    dataset = TestData(test_folder, path_to_classes_json, annotations_file=annotations_file)
    dl_test = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    classes, class_to_idx = DatasetGTSRB.get_classes(path_to_classes_json)
    model.eval()
    for x, path, y in dl_test:
        predicted_cls = model.predict(x)[0]
        results.append({"Path": path[0], "ClassId": classes[predicted_cls]})
    return results


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    function for testing model's performance
    :param model
    :param test_folder: path to test folder
    :param annotations_file: path to .csv file with annotations (target)
    :returns accuracy for all signs, Recall for rare signs, Recall for frequent signs
    """

    with open(path_to_classes_json) as file:
        dct = json.load(file)

    results = apply_classifier(model, test_folder, path_to_classes_json, annotations_file)
    test_answers = pd.read_csv(annotations_file)
    total_acc, rare_recall, freq_recall = 0, 0, 0
    total_rare = 0
    for el in results:
        fn = el["Path"]
        predicted_cls = el["ClassId"]
        real_cls = str(test_answers[test_answers["Path"] == fn]["ClassId"].iloc[0])
        if dct[real_cls]["type"] == "rare":
            total_rare += 1
        if real_cls == predicted_cls:
            total_acc += 1
            if dct[real_cls]["type"] == "rare":
                rare_recall += 1
            else:
                freq_recall += 1
    total_freq = len(results) - total_rare

    return total_acc / len(results), rare_recall / total_rare, freq_recall / total_freq


def train_synt_classifier(train_path, syntdata_path, output_path, classes_json="classes.json", epoch=2):
    """train simple classifier on a mixture of source data and syntetic data"""

    # create dataset, dataloader
    dataset = DatasetGTSRB(
        [train_path, syntdata_path],
        classes_json
    )
    dl_train = DataLoader(dataset, batch_size=256, shuffle=True)

    # create and train model
    model = SimpleClassifier()
    trainer = pl.Trainer(
        max_epochs=epoch,
        callbacks=[],
        enable_checkpointing=False,
        logger=False,
        accelerator="auto"
    )
    trainer.fit(model, dl_train)

    # save
    torch.save(model.state_dict(), output_path)

    return model


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive Loss for embeddings.
    """

    def __init__(self, margin: float) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, x, y):
        # contrastive loss: x are embeddings (batch_size, emb_dimension)
        x = x.T

        emb_dim = x.shape[1]

        first = torch.tile(x.unsqueeze(dim=0), (emb_dim, 1, 1))
        second = torch.clone(first).permute(2, 1, 0)

        norms = torch.sum((first - second) ** 2, dim=1)

        maxims = torch.maximum(self.margin - norms, torch.zeros_like(norms))
        equalities = (torch.tile(y.unsqueeze(dim=0), (y.shape[0], 1)) -
                      torch.tile(y.unsqueeze(dim=1), (1, y.shape[0])))

        equalities = torch.where(equalities != 0, 0, 1)

        eye = torch.eye(emb_dim).to(DEVICE)
        loss = ((equalities - eye) * norms).sum() / (equalities - eye).sum() + ((1 - equalities) * maxims).sum() / (
                1 - equalities).sum()

        return loss / 2


class EmbeddingNetwork(SimpleClassifier):
    """
    Class for improved custom network with
        NN that generates embeddings,
        Constrastive Loss followed by it
    """

    def __init__(self, *args, train_model=False, **kwargs):
        """
        :param args:
        :param kwargs:
        Create improved custom network with
            nn that generates embeddings
            Constrastive loss followed by it
        """
        super().__init__(*args, **kwargs)

        self.internal_features = 1024

        # set losses
        self.contrastive_loss = ContrastiveLoss(2.0)
        self.contrastive_loss_weight = 0.1
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.classification_loss_weight = 1. - self.contrastive_loss_weight

        # set model for generate embeddings (resnet18 without fc and agvpool)
        if train_model:
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights
            )
        else:
            model = torchvision.models.resnet18(
                weights=None
            )
        model.fc = torch.nn.Identity()
        model.avgpool = torch.nn.Identity()

        # set fc layer
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=self.internal_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=self.internal_features, out_features=CLASSES_CNT),
        )

        # freeze some resnet layers weights
        for child in list(self.model.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False

    def training_step(self, batch, idx):
        x, path, y = batch
        # get embeddings
        embs = self.model(x)

        # get embs loss
        contr_loss_value = self.contrastive_loss(embs, y)
        clf_loss_value = self.classification_loss(embs, y)

        self.log_dict({"contr_loss": contr_loss_value, "ce_loss": clf_loss_value},
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True
                      )

        # return contr_loss_value
        loss = contr_loss_value * self.contrastive_loss_weight + clf_loss_value * self.classification_loss_weight
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return [optim]

    def forward(self, x):
        return self.model(x)


def train_embedding_network(train_path, syntdata_path, output_path, classes_json="classes.json", epochs=7):
    """Training classifier with Contrastive Loss on a mixture of source data and syntetic data"""

    # create dataset, dataloader
    dataset = DatasetGTSRB(
        [train_path, syntdata_path],
        classes_json
    )
    dl_train = DataLoader(
        dataset,
        batch_sampler=BatchSampler(dataset, 64, 4),
        pin_memory=True,
        num_workers=4
    )

    # create and train model
    model = EmbeddingNetwork()
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[],
        enable_checkpointing=False,
        logger=False,
        accelerator="auto"
    )
    trainer.fit(model, dl_train)

    # save
    torch.save(model.state_dict(), output_path)

    return model


class KnnModel(torch.nn.Module):
    """
    Model with K-NN head
    :param n_neighbors: num of neighbours for K-NN head
    """

    def __init__(self, n_neighbors):
        super().__init__()
        self.n_neighbors = n_neighbors

    def load_nn(self, nn_weights_path):
        """
        Funciton for loading nn.
        :param nn_weights_path: path to improved model
        """
        self.model = EmbeddingNetwork()
        self.model.load_state_dict(torch.load(nn_weights_path))
        self.model.eval()

    def save_head(self, output_path):
        # Its important to use binary mode
        knnPickle = open(output_path, "wb")

        # source, destination
        pickle.dump(self.k_nn, knnPickle)

        # close the file
        knnPickle.close()

    def train_head(self, synt_data_path, classes_json_path, examples_per_classes=20):
        # define k_nn
        self.k_nn = KNeighborsClassifier(n_neighbors=1)

        # train
        dataset = DatasetGTSRB([synt_data_path], classes_json_path)
        sampler = SamplerForKNN(dataset, examples_per_classes)
        dl_train = DataLoader(
            dataset,
            sampler=sampler,
        )
        prepared_index = np.zeros(
            (len(dl_train), CLASSES_CNT)
        )
        cls_indexes = np.zeros(len(dl_train))
        for i, batch in enumerate(dl_train):
            img, pathes, cls_idx = batch
            # get embeddings
            emb = self.model(img)
            emb = torch.nn.functional.normalize(emb)
            # to CPU and to NumPy ndarray
            emb = emb.detach().cpu().numpy()
            prepared_index[i] = emb
            cls_indexes[i] = cls_idx

        # fit K_NN
        self.k_nn.fit(prepared_index, cls_indexes)

    def load_head(self, knn_path):
        """
        Function for loading K-NN weights
        :param knn_path: path to file knn_model.bin
        """
        self.k_nn = pickle.load(open(knn_path, "rb"))

    def predict(self, imgs):
        """
        Function that predicts classes of images
        :param imgs: batch with images
        """
        features = self.model(imgs).detach().cpu().numpy()
        features = features / np.linalg.norm(features, axis=1)[:, None]
        knn_pred = self.k_nn.predict(features)
        return knn_pred.astype(int)

    def forward(self, imgs):
        return self.predict(imgs)


def train_model_with_knn(nn_weights_path, synt_data_path, output_path, classes_json_path, examples_per_class=20):
    """
    Function for training KNN head of embedding network
    :param nn_weights_path: path to .pth of embedding network
    :param synt_data_path: path to synthetic samples
    :param output_path: path to save trained KNN
    :param classes_json_path: path to classes.json
    :param examples_per_class: num of elements of each class to be used in KNN training
    """
    model = KnnModel(n_neighbors=1)
    model.load_nn(nn_weights_path)
    model.train_head(synt_data_path,
                     classes_json_path,
                     examples_per_classes=examples_per_class)
    model.save_head(output_path)
