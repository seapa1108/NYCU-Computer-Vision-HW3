import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from utils import encode_mask
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes):
    m = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, num_classes
    )
    m.roi_heads.mask_predictor = MaskRCNNPredictor(
        m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes
    )
    return m


@torch.inference_mode()
def run_inference(
    data_root: Path,
    ckpt: Path,
    out_json: Path,
    num_classes: int,
    score_thresh: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    tf = T.ToTensor()
    with open(data_root / "test_image_name_to_ids.json") as f:
        infos = json.load(f)

    results = []
    for info in infos:
        img_path = data_root / "test_release" / info["file_name"]
        img_tensor = tf(Image.open(img_path).convert("RGB")).to(device)
        output = model([img_tensor])[0]
        for box, score, label, mask_logit in zip(
            output["boxes"],
            output["scores"],
            output["labels"],
            output["masks"],
        ):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            rle = encode_mask(
                (mask_logit[0] > 0.5).cpu().numpy().astype(np.uint8)
            )
            results.append(
                {
                    "image_id": info["id"],
                    "bbox": bbox,
                    "score": float(score),
                    "category_id": int(label),
                    "segmentation": rle,
                }
            )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    data_root = Path("./hw3-data-release")
    checkpoint = data_root / "maskrcnn_epoch30.pth"
    output_json = Path("./submission.json")
    num_classes = 5
    score_thresh = 0.05

    run_inference(
        data_root, checkpoint, output_json, num_classes, score_thresh
    )
