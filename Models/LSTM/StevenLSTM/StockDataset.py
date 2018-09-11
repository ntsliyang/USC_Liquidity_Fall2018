import torch
import pandas
import numpy
from torch.utils.data import Dataset

# "Author: Wen Tao (Steven) Zhao"
# WARNING: INCOMPLETE

# Stock dataset for Pytorch
class StockDataset(Dataset):
    """
        Initializes the StockDataset class.

        Input:
            csv_file (string) : Path to the csv file

    """
    def __init__(self, csv_file):
        self.data = pandas.read_csv(csv_file)
        # This is two values! Access one or the other with
        # self.data.values[0][0] or [0][1]
        # print(self.data.values[0])
    """
        Return the csv data length

    """
    def __len__(self):
        return len(self.data)
    """
        Return the csv data length

    """
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    stockData = StockDataset('../../Dataset/Microsoft Share Volume Monthly.csv')

# Run main function. Used purely for testing.
if __name__ == '__main__':
    main()
