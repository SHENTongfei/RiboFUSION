# -*- coding: utf-8 -*-
"""D1 — PocketReaderHead: anchor-resolved peptide-HLA compatibility head.

FIELD-BIASED (domain) architecture, not a generic ML tweak.

Immunology premise (Rammensee 1995; Abelin 2017; Sarkizova 2020):
  MHC-I presentation is decided by a handful of *anchor positions* on the
  peptide (P2 -> B pocket, C-terminus/P-omega -> F pocket) docking into the
  allele-specific binding groove. Matched hard negatives ("non-binders but
  sequence-similar") differ from true binders at exactly one anchor residue,
  so generic whole-sequence semantics cannot separate them — but anchor
  chemistry can. This is precisely the weak spot of pure-PLM representations
  (see H1 §12.3.1 factor 3).

Design (everything implementable from the 37-aa allele_sequences.csv grooves
and raw peptide sequences — no fabricated structural indices):
  1. Peptide is split by *anchor position* (P2, C-term = anchors; rest = body):
       - P2        -> anchor token 1   (domain: defines B-pocket fit)
       - C-term    -> anchor token 2   (domain: defines F-pocket fit)
       - body      -> pooled body
  2. Groove (37 aa) is encoded per-position. Each anchor attends over groove
     positions with learned attention (query = anchor emb, keys = groove embs).
     The resulting attention weights are a *data-driven localization of the
     B/F pocket residues* — we do NOT hard-code structural indices.
  3. Compatibility = MLP( P2_anchor ⊙ B_pocket_context,
                          Cterm_anchor ⊙ F_pocket_context,
                          pep_body, groove_pooled )
  This head can run on its own (anchor-chemistry tower) or be appended as an
  extra tower inside GatedReader. The residual-covariance diversity loss (H1
  §12.2.2) keeps it from cloning the PLM towers.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA_20 = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {c: i for i, c in enumerate(AA_20)}


def encode_residues(seqs, max_len):
    """(N, max_len) one-hot indices; pad 0. seqs: list of AA strings."""
    N = len(seqs)
    X = np.zeros((N, max_len, len(AA_20) + 1), dtype=np.float32)  # +1 for pad
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:max_len]):
            idx = AA2I.get(c)
            if idx is not None:
                X[i, j, idx + 1] = 1.0   # 0 = pad
    return torch.from_numpy(X)


class PocketReaderHead(nn.Module):
    """Anchor-resolved peptide-HLA co-fitness head.

    Args:
      d_model: per-residue embedding dim
      n_heads: attention heads for pocket localization
      hid: MLP hidden size
      max_pep: max peptide length (15 to match routeb encoding)
      max_groove: groove length (37 to match allele_sequences.csv)
    """
    def __init__(self, d_model=64, n_heads=2, hid=256, max_pep=15, max_groove=37):
        super().__init__()
        self.max_pep = max_pep
        self.max_groove = max_groove
        self.pep_emb = nn.Linear(len(AA_20) + 1, d_model)
        self.grv_emb = nn.Linear(len(AA_20) + 1, d_model)
        # position-type embedding for peptide (0=body,1=P2-anchor,2=Cterm-anchor)
        self.pos_type = nn.Embedding(3, d_model)
        # per-anchor pocket attention (data-driven B/F pocket localization)
        self.pocket_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.compat = nn.Sequential(
            nn.Linear(d_model * 4, hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        # gate to decide how much the domain head matters vs PLM towers
        self.gate = nn.Sequential(nn.Linear(d_model * 4, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def _anchor_split(self, pep_feat, lens):
        """Split peptide embeddings into P2-anchor, Cterm-anchor, body pooled."""
        # pep_feat: (N, L, D) ; lens: (N,)
        N, L, D = pep_feat.shape
        dev = pep_feat.device
        result = {"p2": torch.zeros(N, D, device=dev),
                  "cterm": torch.zeros(N, D, device=dev),
                  "body": torch.zeros(N, D, device=dev)}
        for i in range(N):
            L_i = int(lens[i].item())
            if L_i >= 2:
                result["p2"][i] = pep_feat[i, 1]           # P2 (index 1)
                result["cterm"][i] = pep_feat[i, L_i - 1]  # C-terminal anchor
                if L_i > 2:
                    result["body"][i] = pep_feat[i, 2:L_i - 1].mean(0)
        return result

    def forward(self, pep_seqs, groove_seqs):
        """pep_seqs: list[str]; groove_seqs: list[str]. Returns (logit, attn)."""
        N = len(pep_seqs)
        lens = torch.tensor([min(len(s), self.max_pep) for s in pep_seqs],
                            dtype=torch.long)
        # peptide: embed + position-type tagging
        pep_x = encode_residues(pep_seqs, self.max_pep).to(next(self.parameters()).device)
        pep_e = self.pep_emb(pep_x)                       # (N, L, D)
        # position-type: mark P2 (index1) and C-term (index len-1) as anchors
        pos = torch.zeros((N, self.max_pep), dtype=torch.long, device=pep_e.device)
        for i in range(N):
            L_i = int(lens[i].item())
            if L_i >= 2:
                pos[i, 1] = 1
                pos[i, L_i - 1] = 2
        pep_e = pep_e + self.pos_type(pos)                # inject domain position prior
        a = self._anchor_split(pep_e, lens)

        # groove: embed per-position
        grv_x = encode_residues(groove_seqs, self.max_groove).to(pep_e.device)
        grv_e = self.grv_emb(grv_x)                       # (N, G, D)

        # per-anchor pocket localization via attention over groove positions
        b_ctx, B_attn = self.pocket_attn(a["p2"].unsqueeze(1), grv_e, grv_e)
        f_ctx, F_attn = self.pocket_attn(a["cterm"].unsqueeze(1), grv_e, grv_e)
        # compatibility + gate
        grv_pooled = grv_e.mean(1)
        z = torch.cat([a["p2"] * b_ctx.squeeze(1),
                       a["cterm"] * f_ctx.squeeze(1),
                       a["body"], grv_pooled], dim=1)
        logit = self.compat(z).squeeze(1)
        g = torch.sigmoid(self.gate(z)).squeeze(1)
        return logit, g, {"B_attn": B_attn, "F_attn": F_attn}


if __name__ == "__main__":
    torch.manual_seed(0)
    head = PocketReaderHead()
    pep = ["MIPFTSDWV", "SIINFEKL", "RSAGPRPAL", "LLGACRTL", "AIMPLDEI"]
    grv = ["YFAMYGEKVAHTHVDTLYGVRYDHYYTWAVLAYTWYA"] * len(pep)
    logit, g, attn = head(pep, grv)
    print("D1 smoke OK: logit", tuple(logit.shape), "gate", tuple(g.shape))
    print("anchored logit range:", round(float(logit.min()), 3),
          round(float(logit.max()), 3))
    print("B-pocket attention weights (per-sample argmax groove pos):",
          attn["B_attn"].squeeze(1).argmax(-1).tolist())
    print("F-pocket attention weights (per-sample argmax groove pos):",
          attn["F_attn"].squeeze(1).argmax(-1).tolist())