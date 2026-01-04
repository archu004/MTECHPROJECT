import torch

WEIGHTS_PATH = "D:\\MTECHMODELS\\modelfromarchupersonal.pth"
NUM_CLASSES = 7
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIDPOINTS = torch.tensor([5., 15., 25., 35., 45., 55., 70.])
AGE_LABELS = ["1-10","11-20","21-30","31-40","41-50","51-60","61+"]
