"""Evaluation metrics for map matching."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def row_metrics(y, p, thr=0.5):
    """Binary candidate-classification metrics (rows = observation/candidate pairs)."""
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_accuracy(df, prob_col="prob"):
    """Observation-level Top-1: argmax candidate per point == reference arc.
    Points whose reference arc is not in the candidate set count as errors."""
    best = df.loc[df.groupby(["trip_id", "point"])[prob_col].idxmax()]
    return float(best["label"].mean()) if len(best) else 0.0


def route_continuity(comp_seq, net):
    """Fraction of transitions between consecutive distinct predicted arcs that are
    physically connected (end node of a == start node of b)."""
    if len(comp_seq) < 2:
        return 1.0
    a = np.asarray(comp_seq[:-1]); b = np.asarray(comp_seq[1:])
    return float(np.mean(net.arcs[a, 1] == net.arcs[b, 0]))


def route_overlap(pred_route, ref_route, net):
    """Sequence-level metrics between predicted and reference arc sequences.

    * arc precision / recall / F1 on the sets of arcs
    * length-weighted precision / recall / F1 (metres of road correctly matched)
    """
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    inter = P & R
    prec = len(inter) / len(P) if P else 0.0
    rec = len(inter) / len(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    L = net.length
    lp = L[list(P)].sum() if P else 0.0
    lr = L[list(R)].sum() if R else 0.0
    li = L[list(inter)].sum() if inter else 0.0
    lprec = li / lp if lp else 0.0
    lrec = li / lr if lr else 0.0
    lf1 = 2 * lprec * lrec / (lprec + lrec) if lprec + lrec else 0.0
    return {"arc_precision": prec, "arc_recall": rec, "arc_f1": f1,
            "len_precision": lprec, "len_recall": lrec, "len_f1": lf1}


def point_accuracy(pred, labels):
    return float(np.mean(np.asarray(pred) == np.asarray(labels)))
