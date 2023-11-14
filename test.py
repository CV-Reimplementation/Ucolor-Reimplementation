import warnings
import os

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_testing_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

os.makedirs('result', exist_ok=True)

def test():
    accelerator = Accelerator()

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    test_dataset = get_testing_data(val_dir, opt.MODEL.INPUT, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': False})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, test_loader = accelerator.prepare(model, test_loader)

    model.eval()

    for _, test_data in enumerate(tqdm(test_loader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        dep = test_data[1].contiguous()

        with torch.no_grad():
            res = model(inp, dep)

        save_image(res, os.path.join(os.getcwd(), "result", test_data[2][0]))


if __name__ == '__main__':
    test()
