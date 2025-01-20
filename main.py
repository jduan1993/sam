import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

sam = sam_model_registry["vit_b"](checkpoint="model/sam_vit_b_01ec64.pth").to(device="cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("image.jpg")
h, w = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(w / 100, h / 100))
plt.imshow(image)
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("orig_image.jpg")

masks = mask_generator.generate(image)
print(len(masks))
for m in masks:
    print(m.keys())
    print(m['segmentation'])
    print(m['area'])
    print(m['bbox'])
    print(m['predicted_iou'])
    print(m['point_coords'])
    print(m['crop_box'])
plt.figure(figsize=(w / 100, h / 100))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("masked_image.jpg")