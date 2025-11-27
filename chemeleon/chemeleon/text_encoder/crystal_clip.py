from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from transformers import BertModel, BertTokenizer

from chemeleon.text_encoder import MODEL_NAMES
from chemeleon.modules.cspnet import CSPNet
from chemeleon.utils.scatter import scatter_mean, scatter_sum

class GraphEmbeddingPredictor(nn.Module):
    """
    Predicts a graph-style embedding from the text embedding.
    Here we take the projected text embedding (clip_dim) as input and
    output a vector in the same space (clip_dim).
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        z = self.net(x)
        return F.normalize(z, dim=-1)


class EmbeddingFusion(nn.Module):
    """
    Fuses the original text embedding and the GEP embedding into a combined
    stoichiometry- and structure-aware embedding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, z_text: torch.Tensor, z_gep: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_text, z_gep], dim=-1)  # [B, 2*dim]
        z = self.proj(x)
        return F.normalize(z, dim=-1)


class CrystalClip(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters(_config)
        self.clip_dim = _config["clip_dim"]
        self.label_smoothing = _config["label_smoothing"]
        # text encoder
        self.text_encoder_name = _config["text_encoder"]
        self.max_text_len = _config["max_text_len"]
        self.text_embed_dim = _config["text_embed_dim"]
        assert (
            self.text_encoder_name in MODEL_NAMES
        ), f"Invalid model name. Must be one of {MODEL_NAMES}"
        self.tokenizer = BertTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = BertModel.from_pretrained(self.text_encoder_name).to(
            self.device
        )
        self.text_encoder.train()

        # graph encoder
        _config["time_dim"] = 0
        _config["text_dim"] = 0
        assert _config["time_dim"] == 0 and _config["text_dim"] == 0
        self.graph_encoder = CSPNet(
            hidden_dim=_config["hidden_dim"],
            time_dim=_config["time_dim"],
            text_dim=_config["text_dim"],
            num_layers=_config["num_layers"],
            max_atoms=_config["max_atoms"],
            act_fn=_config["act_fn"],
            dis_emb=_config["dis_emb"],
            num_freqs=_config["num_freqs"],
            edge_style=_config["edge_style"],
            cutoff=_config["cutoff"],
            max_neighbors=_config["max_neighbors"],
            ln=_config["ln"],
            ip=_config["ip"],
            smooth=_config["smooth"],
            pred_atom_types=_config["pred_atom_types"],
        )
        self.graph_pooling = _config["graph_pooling"]
        self.graph_embed_dim = _config["hidden_dim"]
        if self.graph_pooling == "mean":
            self.pool_fn = scatter_mean
        elif self.graph_pooling == "sum":
            self.pool_fn = scatter_sum
        # projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.text_embed_dim),
            nn.LayerNorm(self.text_embed_dim),
            nn.GELU(),
            nn.Linear(self.text_embed_dim, self.clip_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_embed_dim, self.graph_embed_dim),
            nn.LayerNorm(self.graph_embed_dim),
            nn.GELU(),
            nn.Linear(self.graph_embed_dim, self.clip_dim),
        )

        # optimizer
        self.lr = _config["lr"]
        self.graph_encoder_lr = _config["graph_encoder_lr"]
        self.text_encoder_lr = _config["text_encoder_lr"]
        self.weight_decay = _config["weight_decay"]
        self.patience = _config["patience"]
                # GEP + fusion for stoichiometry-aware conditioning
        self.use_gep = _config.get("use_gep", False)
        self.gep_hidden_dim = _config.get("gep_hidden_dim", self.clip_dim)
        self.lambda_gep = _config.get("lambda_gep", 0.1)

        if self.use_gep:
            self.gep = GraphEmbeddingPredictor(
                in_dim=self.clip_dim,
                hidden_dim=self.gep_hidden_dim,
                out_dim=self.clip_dim,
            )
            self.fusion = EmbeddingFusion(dim=self.clip_dim)


    def get_text_embeds(self, text: List[str]):
        tokenized = self.tokenizer.batch_encode_plus(
            text,
            padding="longest",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",  # Returns torch.tensor instead of python integers
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        class_token_embeds = outputs.last_hidden_state[:, 0, :]  # [B, text_embed_dim]
        text_embeds = self.text_proj(class_token_embeds)  # [B, clip_dim]
        return text_embeds

    def get_graph_embeds(self, batch: Batch):
        outputs = self.graph_encoder(
            t=None,
            atom_types=batch.atom_types,
            frac_coords=batch.frac_coords,
            lattices=batch.lattices,
            num_atoms=batch.natoms,
            node2graph=batch.batch,
        )
        node_features = outputs.node_features  # [B_n, hidden_dim]
        graph_features = self.pool_fn(
            node_features, batch.batch, dim=0
        )  # [B, hidden_dim]
        graph_embeds = self.graph_proj(graph_features)  # [B, clip_dim]
        return graph_embeds

    def forward(self, batch: Batch):
        # base text + graph embeddings in CLIP space
        text_embeds = self.get_text_embeds(batch.text)      # [B, clip_dim]
        graph_embeds = self.get_graph_embeds(batch)         # [B, clip_dim]

        if self.use_gep:
            gep_embeds = self.gep(text_embeds)              # [B, clip_dim]
            combined_embeds = self.fusion(text_embeds, gep_embeds)  # [B, clip_dim]
        else:
            gep_embeds = None
            combined_embeds = text_embeds

        # we return combined (used for contrastive), the graph, and the raw gep
        return combined_embeds, graph_embeds, gep_embeds


    def compute_contrastive_loss(
        self, text_embeds: torch.Tensor, graph_embeds: torch.Tensor
    ):
        # gather all embeddings
        all_text_embeds = self.all_gather(text_embeds, sync_grads=True).view(
            -1, self.clip_dim
        )  # [B * k, clip_dim]
        all_graph_embeds = self.all_gather(graph_embeds, sync_grads=True).view(
            -1, self.clip_dim
        )  # [B * k, clip_dim]
        # get targets
        graph_similarity = all_graph_embeds @ all_graph_embeds.T  # [B * k, B * k]
        text_similarity = all_text_embeds @ all_text_embeds.T  # [B * k, B * k]
        all_targets = F.softmax((graph_similarity + text_similarity) / 2, dim=-1)
        # get logits
        all_logits = all_text_embeds @ all_graph_embeds.T  # [B * k, B * k]
        # calculate loss
        graph_loss = F.cross_entropy(
            all_logits.T,
            all_targets.argmax(dim=-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # [B * k]
        text_loss = F.cross_entropy(
            all_logits,
            all_targets.argmax(dim=0),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # [B * k]
        loss = (graph_loss + text_loss) / 2  # [B * k]
        loss = loss.mean()
        return loss

    def training_step(self, batch: Batch, *args, **kwargs):
        text_embeds, graph_embeds, gep_embeds = self.forward(batch)

        # main CLIP-style loss between combined embedding and graph embedding
        loss_clip = self.compute_contrastive_loss(text_embeds, graph_embeds)

        # optional auxiliary: encourage GEP to approximate graph embedding
        if self.use_gep and gep_embeds is not None and self.lambda_gep > 0.0:
            loss_gep = F.mse_loss(gep_embeds, graph_embeds)
            loss = loss_clip + self.lambda_gep * loss_gep
            self.log("train/loss_clip", loss_clip)
            self.log("train/loss_gep", loss_gep)
        else:
            loss = loss_clip

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Batch, *args, **kwargs):
        text_embeds, graph_embeds, gep_embeds = self.forward(batch)
        loss_clip = self.compute_contrastive_loss(text_embeds, graph_embeds)
        if self.use_gep and gep_embeds is not None and self.lambda_gep > 0.0:
            loss_gep = F.mse_loss(gep_embeds, graph_embeds)
            loss = loss_clip + self.lambda_gep * loss_gep
            self.log("val/loss_gep", loss_gep)
        else:
            loss = loss_clip
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch: Batch, *args, **kwargs):
        text_embeds, graph_embeds, gep_embeds = self.forward(batch)
        loss_clip = self.compute_contrastive_loss(text_embeds, graph_embeds)
        if self.use_gep and gep_embeds is not None and self.lambda_gep > 0.0:
            loss_gep = F.mse_loss(gep_embeds, graph_embeds)
            loss = loss_clip + self.lambda_gep * loss_gep
            self.log("test/loss_gep", loss_gep)
        else:
            loss = loss_clip
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        parameters = [
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {"params": self.graph_encoder.parameters(), "lr": self.graph_encoder_lr},
            {"params": self.text_proj.parameters(), "lr": self.lr},
            {"params": self.graph_proj.parameters(), "lr": self.lr},
        ]
        if getattr(self, "use_gep", False):
            parameters.append({"params": self.gep.parameters(), "lr": self.lr})
            parameters.append({"params": self.fusion.parameters(), "lr": self.lr})

        optimizer = torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", min_lr=1e-6, factor=0.8, patience=self.patience
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning rate",
            "monitor": "val/loss",
        }

        return ([optimizer], [lr_scheduler])
    
    @torch.no_grad()
    def get_conditioning_embeds(self, texts: List[str]) -> torch.Tensor:
        # base CLIP text embeddings (already projected to clip_dim)
        z_text = self.get_text_embeds(texts)  # [B, clip_dim]

        if getattr(self, "use_gep", False) and hasattr(self, "gep"):
            z_gep = self.gep(z_text)
            z_combined = self.fusion(z_text, z_gep)
            return z_combined  # [B, clip_dim], normalized by fusion
        else:
            # fall back to plain CLIP text embedding
            return z_text


