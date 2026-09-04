from __future__ import annotations

import copy
import time

import numpy as np

from asdgnn.train.metrics import compute_all
from asdgnn.utils.logging import get_logger
from asdgnn.utils.seed import set_seed

log = get_logger(__name__)

# Hyperparameters that actually work on this data (N~871, very high dim):
DEFAULTS = dict(
    lr=1e-3, weight_decay=5e-4, epochs=200, min_epochs=30, patience=20,
    batch_size=32, dropout=0.5, class_weight=True, monitor="auc",
)


def _device(name: str | None = None):
    import torch

    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_weights(y, device):
    import torch

    y = np.asarray(y)
    counts = np.bincount(y, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float, device=device)


def _probs(logits):
    import torch

    return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


def train_graph_level(model, train_graphs, val_graphs, test_graphs, cfg: dict,
                      seed: int = 0) -> dict:
    """Train one fold of a graph-level model. Returns metrics + per-sample test
    probabilities — without the probabilities you cannot run DeLong later, and
    re-running 1000 folds because you forgot is a miserable afternoon."""
    import torch
    from torch_geometric.loader import DataLoader

    from asdgnn.utils.seed import seed_worker

    c = {**DEFAULTS, **cfg}
    set_seed(seed)
    device = _device(c.get("device"))
    model = model.to(device)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_graphs, batch_size=c["batch_size"], shuffle=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_graphs, batch_size=256) if val_graphs else None
    test_loader = DataLoader(test_graphs, batch_size=256)

    y_train = [int(d.y.item()) for d in train_graphs]
    weight = _class_weights(y_train, device) if c["class_weight"] else None
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=10)

    best = {"score": -np.inf, "state": None, "epoch": -1}
    history, t0 = [], time.time()

    for epoch in range(c["epochs"]):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(model(batch), batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.detach().item() * batch.num_graphs
        train_loss = total / max(1, len(train_graphs))

        if val_loader is not None:
            yv, pv = _predict(model, val_loader, device)
            vm = compute_all(yv, pv)
            score = vm.get(c["monitor"]) or vm["bal_acc"]
            sched.step(score)
            history.append({"epoch": epoch, "train_loss": train_loss, **vm})
            if score > best["score"]:
                best = {"score": score, "state": copy.deepcopy(model.state_dict()),
                        "epoch": epoch}
            if epoch >= c["min_epochs"] and epoch - best["epoch"] >= c["patience"]:
                log.debug("early stop at epoch %d (best %d)", epoch, best["epoch"])
                break
        else:
            history.append({"epoch": epoch, "train_loss": train_loss})

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    y_true, y_prob = _predict(model, test_loader, device)
    return {
        "metrics": compute_all(y_true, y_prob),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "sub_ids": [int(d.sub_id.item()) for d in test_graphs],
        "best_epoch": best["epoch"],
        "n_epochs_run": len(history),
        "runtime_sec": round(time.time() - t0, 2),
        "history": history if c.get("save_history") else history[-1:],
    }


def _predict(model, loader, device):
    import torch

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ps.append(_probs(model(batch)))
            ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_node_level(model, data, cfg: dict, seed: int = 0) -> dict:
    """Transductive training on the population graph. `data.y[test_mask]` is never
    touched by the loss — that invariant is the whole reason this is not leakage."""
    import torch

    c = {**DEFAULTS, **cfg}
    set_seed(seed)
    device = _device(c.get("device"))
    model, data = model.to(device), data.to(device)

    weight = (_class_weights(data.y[data.train_mask].cpu().numpy(), device)
              if c["class_weight"] else None)
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

    best = {"score": -np.inf, "state": None, "epoch": -1}
    t0 = time.time()
    for epoch in range(c["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(data)
        loss = crit(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
        vm = compute_all(data.y[data.val_mask].cpu().numpy(),
                         _probs(logits)[data.val_mask.cpu().numpy()])
        score = vm.get(c["monitor"]) or vm["bal_acc"]
        if score > best["score"]:
            best = {"score": score, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch}
        if epoch >= c["min_epochs"] and epoch - best["epoch"] >= c["patience"]:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        prob = _probs(model(data))
    te = data.test_mask.cpu().numpy()
    y_true = data.y.cpu().numpy()[te]
    y_prob = prob[te]
    return {
        "metrics": compute_all(y_true, y_prob),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "best_epoch": best["epoch"],
        "runtime_sec": round(time.time() - t0, 2),
    }
