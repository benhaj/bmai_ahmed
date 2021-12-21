import torch
import pandas as pd
import cv2
from prepare_dataset import *


# In[138]:



model_type1 = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type2 = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas1 = torch.hub.load("intel-isl/MiDaS", model_type1)
midas2 = torch.hub.load("intel-isl/MiDaS", model_type2)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas1.to(device)
midas1.eval()
midas2.to(device)
midas2.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform


data = pd.read_csv('/hdd/data/bmai_clean/full_cambodge_data.csv')
paths = data.img.values[:10]
paths = ["/hdd/data/"+path.replace('data','bmai_clean',1) for path in paths]
paths = [prepare_new_path(path,386) for path in paths]
imgs= [cv2.imread(filename) for filename in paths]
imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

with torch.no_grad():
    for i in range(len(imgs)):
        img = imgs[i].copy()
        input_batch = transform(img).to(device)
        prediction1 = midas1(input_batch)
        prediction2 = midas2(input_batch)
        
        prediction1 = torch.nn.functional.interpolate(
            prediction1.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        prediction2 = torch.nn.functional.interpolate(
            prediction2.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output1 = prediction1.cpu().numpy()
        output2 = prediction2.cpu().numpy()
        
        cv2.imwrite(f'midas_{model_type1}_{i}',output1)
        cv2.imwrite(f'midas_{model_type2}_{i}',output2)
        