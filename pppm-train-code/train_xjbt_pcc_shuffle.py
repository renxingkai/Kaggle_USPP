import gc
import os
import time
import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
# -------------------------------------------------------------------------------------------------


class Setting:
    seed = 17
    n_splits = 5
    train_folds = [0, 1, 2, 3, 4]
    valid_freq = 500

    pretrained_model_path = "../pretrain/deberta-v3-large"
    output_model_path = "../models/xjbt_pcc_shf"

    max_len = 180
    epochs = 5
    lr = 2e-5
    train_batch_size = 32
    valid_batch_size = 128
    apex = True
    gradient_accumulation_steps = 1
    num_warmup_steps = 100

    scheduler = "cosine"
    batch_scheduler = True
    num_cycles = 0.5

    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    weight_decay = 0.01
    max_grad_norm = 1000


cfg = Setting()
# -------------------------------------------------------------------------------------------------


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use {device}")
# -------------------------------------------------------------------------------------------------


dtype = {
    "id": "str",
    "anchor": "str",
    "target": "str",
    "score": "float32",
}

titles = pd.read_csv("../input/cpc-codes/titles.csv")
first_class_mapping = titles.loc[titles["code"].isin(["A", "B", "C", "D", "E", "F", "G", "H", "Y"]), ["code", "title"]]
first_class_mapping = dict(zip(first_class_mapping["code"], first_class_mapping["title"]))

train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv", dtype=dtype)
train = train.merge(titles, left_on="context", right_on="code")

train["section_name"] = train["section"].map(first_class_mapping)
train["context_texts"] = train["section_name"] + ". " + train["title"]

train["score_map"] = train["score"].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
train["anchor_map"] = LabelEncoder().fit_transform(train["anchor"])

kfold = GroupKFold(n_splits=cfg.n_splits)
for idx, (_, valid_idx) in enumerate(kfold.split(train, groups=train["anchor_map"])):
    train.loc[valid_idx, "fold"] = idx

train["fold"] = train["fold"].astype("int")

print(f"train shape: {train.shape}")
print("fold data distribution")
print(train.groupby("fold").size())

# target sequence
group = train.groupby(["anchor", "context"]).agg({"target": list}).reset_index()
group.rename(columns={"target": "anchor_target"}, inplace=True)
train = train.merge(group, how="left", on=["anchor", "context"])
# -------------------------------------------------------------------------------------------------


def anchor_target_random_shuffle(target, anchor_target):
    shuffled = [element for element in anchor_target if element != target]
    shuffled = random.sample(shuffled, min(len(shuffled), 29))
    shuffled = shuffled + [target]
    random.shuffle(shuffled)

    return shuffled


class PatentPhraseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.is_train = is_train
        self.tokenizer = tokenizer

        self.anchor = df["anchor"].tolist()
        self.target = df["target"].tolist()
        self.context = df["context_texts"].tolist()
        self.anchor_target = df["anchor_target"].tolist()

        if self.is_train:
            self.score = df["score"].tolist()

    def __getitem__(self, index):
        anchor = self.anchor[index]
        target = self.target[index]
        context = self.context[index]

        anchor_target = self.anchor_target[index]
        anchor_target = ";".join(anchor_target_random_shuffle(target, anchor_target))

        inputs = anchor + "[SEP]" + target + ". " + anchor_target + "[SEP]" + context

        encoded = self.tokenizer(
            inputs,
            max_length=cfg.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
        )

        if self.is_train:
            return {
                "ids": torch.LongTensor(encoded["input_ids"]),
                "masks": torch.LongTensor(encoded["attention_mask"]),
                "labels": torch.FloatTensor([self.score[index]])
            }
        else:
            return {
                "ids": torch.LongTensor(encoded["input_ids"]),
                "masks": torch.LongTensor(encoded["attention_mask"]),
            }

    def __len__(self):
        return len(self.anchor)


class PatentPhraseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(cfg.pretrained_model_path)
        config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7
        })

        self.bert = AutoModel.from_pretrained(cfg.pretrained_model_path, config=config)
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    def forward(self, ids, masks):
        bert_output = self.bert(input_ids=ids, attention_mask=masks)
        last_hidden_states = bert_output.last_hidden_state

        return self.regressor(last_hidden_states[:, 0])

    def init_weights(self):
        torch.nn.init.normal_(self.regressor[1].weight, mean=0, std=0.001)
        torch.nn.init.zeros_(self.regressor[1].bias)


class PCCLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        p1 = y_true - y_true.mean()
        p2 = y_pred - y_pred.mean()

        cov = torch.sum(p1 * p2)
        sigma = (p1.pow(2).sum() * p2.pow(2).sum()).sqrt() + 1e-6

        return 1 - cov / sigma
# -------------------------------------------------------------------------------------------------


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         "lr": encoder_lr, "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if "bert" not in n],
         "lr": decoder_lr, "weight_decay": 0.0}
    ]
    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=num_train_steps, T_mult=2, eta_min=1e-6,
        # )
    return scheduler


def data_fn(fold, data, tokenizer):
    data = data.reset_index(drop=True)

    train = data.loc[data["fold"] != fold]
    valid = data.loc[data["fold"] == fold]

    train_dataset = PatentPhraseDataset(train, tokenizer)
    valid_dataset = PatentPhraseDataset(valid, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, drop_last=True, batch_size=cfg.train_batch_size,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=False, drop_last=False, batch_size=cfg.valid_batch_size,
    )

    return {
        "train_idx": train.index,
        "valid_idx": valid.index,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }
# -------------------------------------------------------------------------------------------------


criterion = PCCLoss()
tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)
score_folds = list()
oof_score = np.zeros(train.shape[0])

for idx in cfg.train_folds:
    net = PatentPhraseModel()
    net.to(device)

    # data preparation
    data_output = data_fn(idx, train, tokenizer)
    train_loader = data_output["train_loader"]
    valid_loader = data_output["valid_loader"]
    # break

    # optimizer
    optimizer_parameters = get_optimizer_params(
        net, encoder_lr=cfg.encoder_lr, decoder_lr=cfg.decoder_lr, weight_decay=cfg.weight_decay
    )
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.encoder_lr)

    # scheduler
    num_train_steps = int(cfg.epochs / 3 * len(train_loader))
    # num_train_steps = int(len(data_output["train_idx"]) / cfg.train_batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    best_score = 0
    print(f"Split: {idx + 1}/{cfg.n_splits}\n")

    print("|  Epoch  |     Step     | Train Loss | Valid Loss | Valid PCC  | Learning Rate  | Elapsed Time |")
    print("| ------- | ------------ | ---------- | ---------- | ---------- | -------------- | ------------ |")
    for e in range(cfg.epochs):
        start = time.perf_counter()
        train_losses = list()

        net.train()
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
        for step, inputs in enumerate(train_loader):
            ids, masks = inputs["ids"].to(device), inputs["masks"].to(device)
            labels = inputs["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=cfg.apex):
                output = net(ids=ids, masks=masks)
                loss = criterion(output.view(-1, 1), labels.view(-1, 1))
            if cfg.gradient_accumulation_steps > 1:
                loss /= cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if cfg.batch_scheduler:
                    scheduler.step()

            train_losses.append(loss.item())

            if (step + 1) % cfg.valid_freq == 0:
                valid_array = list()
                valid_labels = list()
                valid_losses = list()

                net.eval()
                for inputs in valid_loader:
                    ids, masks = inputs["ids"].to(device), inputs["masks"].to(device)
                    labels = inputs["labels"].to(device)
                    with torch.no_grad():
                        output = net(ids=ids, masks=masks)
                        val_loss = criterion(output.view(-1, 1), labels.view(-1, 1))
                    if cfg.gradient_accumulation_steps > 1:
                        val_loss = val_loss / cfg.gradient_accumulation_steps

                    valid_losses.append(val_loss.item())
                    valid_array.append(output.cpu().numpy().ravel())
                    valid_labels.append(labels.cpu().numpy().ravel())

                valid_output = {
                    "valid_losses": np.asarray(valid_losses),
                    "valid_array": np.concatenate(valid_array),
                    "valid_labels": np.concatenate(valid_labels),
                }
                train_output = {
                    "step": step,
                    "train_losses": np.asarray(train_losses),
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                }

                # output train information
                score = stats.pearsonr(valid_output["valid_array"], valid_output["valid_labels"])[0]
                duration = time.perf_counter() - start
                print(
                    "|  {}/{} ".format(e + 1, cfg.epochs).ljust(9),
                    "|  {}/{} ".format(train_output["step"], len(train_loader)).ljust(14),
                    "|   {:.4f}  ".format(train_output["train_losses"].mean()).ljust(12),
                    "|   {:.4f}  ".format(valid_output["valid_losses"].mean()).ljust(12),
                    "|   {:.4f}  ".format(score).ljust(12),
                    "|   {:.4e}  ".format(train_output["scheduler"].get_last_lr()[0]).ljust(16),
                    "|  {:.2f}s ".format(duration).ljust(14),
                    "|"
                )
                if best_score < score:
                    best_score = score
                    oof_score[data_output["valid_idx"]] = valid_output["valid_array"]
                    model_path = os.path.join(cfg.output_model_path, f"{idx + 1}.pth")
                    torch.save(net.state_dict(), model_path)

        train_output = {
            "step": step,
            "train_losses": np.asarray(train_losses),
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

        valid_array = list()
        valid_labels = list()
        valid_losses = list()

        net.eval()
        for inputs in valid_loader:
            ids, masks = inputs["ids"].to(device), inputs["masks"].to(device)
            labels = inputs["labels"].to(device)
            with torch.no_grad():
                output = net(ids=ids, masks=masks)
                val_loss = criterion(output.view(-1, 1), labels.view(-1, 1))
            if cfg.gradient_accumulation_steps > 1:
                val_loss = val_loss / cfg.gradient_accumulation_steps

            valid_losses.append(val_loss.item())
            valid_array.append(output.cpu().numpy().ravel())
            valid_labels.append(labels.cpu().numpy().ravel())

        valid_output = {
            "valid_losses": np.asarray(valid_losses),
            "valid_array": np.concatenate(valid_array),
            "valid_labels": np.concatenate(valid_labels),
        }

        # output train information
        score = stats.pearsonr(valid_output["valid_array"], valid_output["valid_labels"])[0]
        duration = time.perf_counter() - start
        print(
            "|  {}/{} ".format(e + 1, cfg.epochs).ljust(9),
            "|  {}/{} ".format(train_output["step"], len(train_loader)).ljust(14),
            "|   {:.4f}  ".format(train_output["train_losses"].mean()).ljust(12),
            "|   {:.4f}  ".format(valid_output["valid_losses"].mean()).ljust(12),
            "|   {:.4f}  ".format(score).ljust(12),
            "|   {:.4e}  ".format(train_output["scheduler"].get_last_lr()[0]).ljust(16),
            "|  {:.2f}s ".format(duration).ljust(14),
            "|"
        )
        if best_score < score:
            best_score = score
            oof_score[data_output["valid_idx"]] = valid_output["valid_array"]
            model_path = os.path.join(cfg.output_model_path, f"{idx + 1}.pth")
            torch.save(net.state_dict(), model_path)

    print(f"PCC is {best_score:.6f}\n")
    score_folds.append(best_score)

    del data_output, train_output, valid_output, net
    torch.cuda.empty_cache()
    gc.collect()

np.save(os.path.join(cfg.output_model_path, "oof_score.npy"), oof_score)
score_final = stats.pearsonr(oof_score, train["score"])[0]
print(f"PCC oof is: {score_final:.6f}")
# -------------------------------------------------------------------------------------------------


for idx in cfg.train_folds:
    data_output = data_fn(idx, train, tokenizer)
    idxes = data_output["valid_idx"]

    # output train information
    score = stats.pearsonr(oof_score[idxes], train.iloc[idxes]["score"])[0]
    print(f"pcc={score:.4f}, mean={oof_score[idxes].mean():.4f}, std={oof_score[idxes].std():.4f}")

    scaled = MinMaxScaler().fit_transform(oof_score[idxes].reshape(-1, 1)).ravel()
    scaled_score = stats.pearsonr(scaled, train.iloc[idxes]["score"])[0]
    print(f"scaled pcc={scaled_score:.4f}, mean={scaled.mean():.4f}, std={scaled.std():.4f}")

    train.loc[idxes, "oof_score"] = scaled

usecols = ["id", "oof_score"]
train[usecols].to_csv(cfg.output_model_path + "/oof_df.csv", index=False)
scaled_oof_score = stats.pearsonr(train["oof_score"], train["score"])[0]
print(f"PCC oof is: {scaled_oof_score:.6f}")
