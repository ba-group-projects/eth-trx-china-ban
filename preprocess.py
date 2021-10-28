import os
import json
import dateutil
import numpy as np
import pandas as pd
from web3 import Web3
from tqdm import tqdm, tqdm_pandas
from google_drive_downloader import GoogleDriveDownloader as gdd

tqdm.pandas(desc="My progress_apply")


class PreprocessData:
    def __init__(self, start_row: int, end_row: int):
        """
        Specify the start_row and end_row
        :param start_row:
        :param end_row:
        """
        with open("config.json", "r") as f:
            conf = json.load(f)
        self.file_id = conf["raw_data_file_id"]
        self.web3_provider = conf["web3_provider"]
        self.download_data()  # if the file doesn't exist'
        self.start_row = start_row
        self.end_row = end_row
        self.dfm = pd.read_csv("./data/raw_data.csv").iloc[start_row:end_row, :]
        self.w3_list = [Web3(Web3.HTTPProvider(x)) for x in self.web3_provider]
        self.w3_list_length = len(self.w3_list)
        self.nounce = 0  # for using w3_sever

    def clean_and_save_data(self):
        self.add_address_type()
        self.divide_value_by_1e18()
        self.dfm.to_csv(f"./data/cleaned_data{self.start_row}_{self.end_row}.csv")

    def download_data(self):
        if not os.path.exists("./data/raw_data.csv"):
            gdd.download_file_from_google_drive(file_id=self.file_id,
                                                dest_path='./data/raw_data.csv',
                                                unzip=True)
        else:
            print("File already downloaded")

    def parse_datetime(self):
        self.dfm["block_timestamp"] = self.dfm["block_timestamp"].progress_apply(lambda x: dateutil.parser.parse(x))

    def select_data(self, start_time: str, end_time: str):  # TODO
        self.dfm = self.dfm[self.dfm["block_timestamp"].apply(lambda x: x.startswith("2021-09-24"))]

    def divide_value_by_1e18(self) -> None:
        """
        Divide the 'value' columns_by_10^18, because the unit of value is wei. We should transform it into ether.
        """
        self.dfm["value"] = self.dfm["value"].apply(lambda x: int(x) / (10 ** 18) if x is not np.nan else np.nan)

    def _judge_address_type(self, address: str):
        """
        Judge the type of address
        :param address:
        :return: "EDA" or "Contract"
        """
        # Because the time of calling api is limited, so we created a api_token list to avoid the limitation
        index = self.nounce % self.w3_list_length
        res = self.w3_list[index].eth.get_code(self.w3_list[index].toChecksumAddress(address)).hex()
        self.nounce += 1
        if res == "0x":
            return "EOA"
        else:
            return "Contract"

    def _get_address_type(self) -> dict:
        all_address = self.dfm["from_address"].append(self.dfm["to_address"]).unique()

    def add_address_type(self) -> None:  # FIXME add infura
        """
        Define the address type as EOA(owned by a person) or Contact
        """
        print("Start to check the type of from_address")
        self.dfm["from_address_type"] = self.dfm["from_address"].progress_apply(self._judge_address_type)
        print("Start to check the type of to_address")
        self.dfm["to_address_type"] = self.dfm["to_address"].progress_apply(self._judge_address_type)


if __name__ == '__main__':
    preprocess_data = PreprocessData(1, 100)
    preprocess_data.clean_and_save_data()
