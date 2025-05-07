import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
from utils import encode_mask, read_maskfile
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from scipy.ndimage import label as cc_label


def get_transform(train=True):
    return T.Compose([T.ToTensor()])


class TiffMaskDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.sample_dirs = sorted(
            [p for p in Path(root).iterdir() if p.is_dir()]
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sd = self.sample_dirs[idx]
        img = Image.open(sd / "image.tif").convert("RGB")
        masks, labels = [], []
        for mp in sorted(sd.glob("class*.tif")):
            bin_mask = read_maskfile(str(mp)) > 0
            cls = (
                int(mp.stem.replace("class", ""))
                if mp.stem.startswith("class")
                else 1
            )
            labeled, n_inst = cc_label(bin_mask)
            for k in range(1, n_inst + 1):
                inst = labeled == k
                if not inst.any():
                    continue
                ys, xs = np.where(inst)
                if xs.max() == xs.min() or ys.max() == ys.min():
                    continue
                masks.append(inst)
                labels.append(cls)

        if masks:
            masks_np = np.stack(masks)
            boxes = torch.tensor(
                [
                    [xs.min(), ys.min(), xs.max(), ys.max()]
                    for inst in masks
                    for ys, xs in [np.where(inst)]
                ],
                dtype=torch.float32,
            )
            masks_t = torch.tensor(masks_np, dtype=torch.uint8)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            w, h = img.size
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            labels_t = torch.tensor([], dtype=torch.int64)

        area = (
            ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
            if boxes.numel() > 0
            else torch.tensor([])
        )
        target = {
            "boxes": boxes,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        img_t = self.transforms(img) if self.transforms else img
        return img_t, target


def get_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    mi_f = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mi_f, 256, num_classes)
    return model


def train(
    data_root,
    num_epochs=30,
    batch_size=2,
    lr=0.005,
    num_classes=5,
    val_split=0.2,
    ckpt_every=5,
):
    ds = TiffMaskDataset(
        Path(data_root) / "train", transforms=get_transform(True)
    )
    n_val = max(1, int(len(ds) * val_split))
    train_ds, val_ds = random_split(
        ds,
        [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes).to(device)
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

    metric = MeanAveragePrecision(iou_type="segm", box_format="xyxy")
    hist = {"train_loss": [], "val_loss": [], "train_map": [], "val_map": []}

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, tgts in tqdm(
            train_loader, desc=f"Train {epoch}/{num_epochs}"
        ):
            imgs = [i.to(device) for i in imgs]
            tgts_dev = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            losses = model(imgs, tgts_dev)
            loss = sum(losses.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        hist["train_loss"].append(train_loss)

        # Training mAP
        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, tgts in train_loader:
                imgs = [i.to(device) for i in imgs]
                preds = model(imgs)
                preds_fmt, tgts_fmt = [], []
                for p, t in zip(preds, tgts):
                    preds_fmt.append(
                        {
                            "boxes": p["boxes"].cpu(),
                            "scores": p["scores"].cpu(),
                            "labels": p["labels"].cpu(),
                            "masks": (p["masks"] > 0.5)
                            .squeeze(1)
                            .cpu()
                            .to(torch.uint8),
                        }
                    )
                    tgts_fmt.append(
                        {
                            "boxes": t["boxes"],
                            "labels": t["labels"],
                            "masks": t["masks"],
                        }
                    )
                metric.update(preds_fmt, tgts_fmt)
        train_map = metric.compute()["map"].item()
        hist["train_map"].append(train_map)

        # Validation loss
        model.train()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = [i.to(device) for i in imgs]
                tgts_dev = [
                    {k: v.to(device) for k, v in t.items()} for t in tgts
                ]
                losses = model(imgs, tgts_dev)
                val_running_loss += sum(losses.values()).item()
        val_loss = val_running_loss / len(val_loader)
        hist["val_loss"].append(val_loss)

        # Validation mAP
        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = [i.to(device) for i in imgs]
                preds = model(imgs)
                preds_fmt, tgts_fmt = [], []
                for p, t in zip(preds, tgts):
                    preds_fmt.append(
                        {
                            "boxes": p["boxes"].cpu(),
                            "scores": p["scores"].cpu(),
                            "labels": p["labels"].cpu(),
                            "masks": (p["masks"] > 0.5)
                            .squeeze(1)
                            .cpu()
                            .to(torch.uint8),
                        }
                    )
                    tgts_fmt.append(
                        {
                            "boxes": t["boxes"],
                            "labels": t["labels"],
                            "masks": t["masks"],
                        }
                    )
                metric.update(preds_fmt, tgts_fmt)
        val_map = metric.compute()["map"].item()
        hist["val_map"].append(val_map)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f},"
            f" Train mAP={train_map:.4f}, Val mAP={val_map:.4f}"
        )
        scheduler.step()

        if epoch % ckpt_every == 0 or epoch == num_epochs:
            ckpt_p = Path(data_root) / f"maskrcnn_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_p)
            print("Saved checkpoint:", ckpt_p)

        epochs = list(range(1, epoch + 1))
        plt.figure()
        plt.plot(epochs, hist["val_map"], label="Val mAP")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title("mAP")
        plt.legend()
        plt.savefig(Path(data_root) / "map_curve.png")
        plt.close()

        plt.figure()
        plt.plot(epochs, hist["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(Path(data_root) / "loss_curve.png")
        plt.close()

    torch.save(model.state_dict(), Path(data_root) / "maskrcnn_final.pth")


def inference(
    data_root,
    output_json,
    num_classes=5,
    ckpt="maskrcnn_final.pth",
    score_thresh=0.01,
):
    info_p = Path(data_root) / "test_image_name_to_ids.json"
    with open(info_p) as f:
        infos = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(
        torch.load(Path(data_root) / ckpt, map_location=device)
    )
    model.to(device).eval()
    tf = T.ToTensor()
    results = []
    with torch.no_grad():
        for info in tqdm(infos, desc="Inference"):
            img = Image.open(
                Path(data_root) / "test_release" / info["file_name"]
            ).convert("RGB")
            out = model([tf(img).to(device)])[0]
            for b, s, l, m in zip(
                out["boxes"], out["scores"], out["labels"], out["masks"]
            ):
                if s < score_thresh:
                    continue
                x1, y1, x2, y2 = b.cpu().numpy()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                results.append(
                    {
                        "image_id": info["id"],
                        "bbox": [float(v) for v in bbox],
                        "score": float(s),
                        "category_id": int(l),
                        "segmentation": encode_mask(
                            (m[0] > 0.5).cpu().numpy().astype(np.uint8)
                        ),
                    }
                )
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    root = "./hw3-data-release"
    train(
        root,
        num_epochs=30,
        batch_size=2,
        num_classes=5,
        val_split=0.2,
        ckpt_every=5,
    )
    inference(
        root,
        "submission.json",
        num_classes=5,
        ckpt="maskrcnn_final.pth",
        score_thresh=0.01,
    )
