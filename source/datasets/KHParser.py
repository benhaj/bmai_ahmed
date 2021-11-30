import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd
import cv2
import os


class BmaiDataset(Dataset):
    def __init__(self, root_dir, xls_filepath, image_size):
        """
        Initializes the dataset class which loads the images and targets pairs.
        :param root_dir: location where the images are stored.
        :param xls_filepath: location of the .xlsx file containing the targets.
        :param image_size: dimension of the images.
        """
        self.image_size = image_size
        self.root_dir = root_dir
        self.image_filepaths = self.collect_filepaths()
        self.data = pd.read_excel(xls_filepath)
        print('-> Duplicated subjects: {}'.format(
            self.data[self.data.duplicated('ChildID', keep='first')]['ChildID'].values))
        self.data.drop_duplicates(subset=['ChildID'], keep=False, inplace=True)

    def collect_filepaths(self):
        """
        Collects all image filepaths in the sub-directories of the provided root directory.
        :return: all the found image filepaths as a list.
        """
        filepaths = []
        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                # Get the filepath
                filepath = os.path.join(path, name)

                # Filter non-image files
                if 'jpg' == name.split('.')[-1].lower():
                    # Filter irrelevant images
                    id_dir = filepath.split('/')[-2]
                    if id_dir in name:
                        filepaths.append(filepath)
        return filepaths

    def pad_image(self, image):
        """
        Pads and resize the image s.t. the resulting images keeps the same ratio, but with the desired dimensions.
        :param image: image to be resized and padded.
        :return: resized and padded image.
        """
        h, w, c = image.shape
        max_dim = max(h, w)
        ratio = max_dim / self.image_size
        h_new, w_new = int(h / ratio), int(w / ratio)

        # Initialize the padded image
        padded_image = np.zeros([self.image_size, self.image_size, c])

        # Compute the padding
        up = (self.image_size - h_new) // 2
        down = h_new + up
        left = (self.image_size - w_new) // 2
        right = w_new + left

        # Resize the image
        image = cv2.resize(image, dsize=(w_new, h_new), interpolation=cv2.INTER_CUBIC)

        # Fill the padded image
        padded_image[up: down, left: right, :] = image
        return padded_image.astype(np.uint8)

    def __len__(self):
        """
        Computes the length of the dataset.
        :return: length of the dataset.
        """
        return len(self.image_filepaths)

    def __getitem__(self, index):
        """
        Loads the queried image and its targets (height, weight).
        :param index: index of the image to retrieve.
        :return: image and its targets.
        """
        # Get the image
        image_path = self.image_filepaths[index]
        image = Image.open(image_path)
        image = self.pad_image(np.array(image))

        # Get the targets
        child_id = image_path.split('/')[-2]
        child_row = self.data[self.data['ChildID'] == child_id]
        weight_1, weight_2 = child_row['Weight1_kg'].astype(float), child_row['Weight2_kg'].astype(float)
        height_1, height_2 = child_row['Height1_cm'].astype(float), child_row['Height2_cm'].astype(float)
        weight = np.nanmean([weight_1, weight_2])
        height = np.nanmean([height_1, height_2])
        return image, (weight, height)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    input_dir1 = '/media/thomas/Samsung_T5/bmai/images/KH_September2018'
    input_dir2 = '/media/thomas/Samsung_T5/bmai/images/KH_August2018'
    xlsx_file1 = '/media/thomas/Samsung_T5/bmai/images/KH_September2018/All Photo Anthro Dataset_part 2.xlsx'
    xlsx_file2 = '/media/thomas/Samsung_T5/bmai/images/KH_August2018/KH_Data_180806.xlsx'
    image_size = 224
    dataset = BmaiDataset(input_dir1, xlsx_file1, image_size)
    image, _ = dataset.__getitem__(0)
    plt.imshow(image)
    plt.show()

