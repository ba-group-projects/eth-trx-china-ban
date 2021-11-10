import datetime
import os
import json
import pytz
import dateutil
import multiprocessing as mp
import numpy as np
import pandas as pd
from web3 import Web3
from tqdm import tqdm, tqdm_pandas
from google_drive_downloader import GoogleDriveDownloader as gdd

tqdm.pandas(desc="My progress_apply")


class PreprocessData:  # TODO Split this class into separate classes to decouple it
    def __init__(self):
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
        self.dfm = pd.read_csv("./data/raw_data.csv")
        self.w3_list = [Web3(Web3.HTTPProvider(x)) for x in self.web3_provider]
        self.w3_list_length = len(self.w3_list)
        self.nounce = 0  # for using w3_sever

    def select_data_by_time(self, start_time: datetime.datetime = datetime.datetime(2021, 9, 24, 1, 0, 0, 0, pytz.UTC),
                            end_time: datetime.datetime = datetime.datetime(2021, 9, 24, 13, 0, 0, 0, pytz.UTC)):
        self.dfm["block_timestamp"] = pd.to_datetime(self.dfm["block_timestamp"])
        self.dfm = self.dfm[(start_time <= self.dfm["block_timestamp"]) & (self.dfm["block_timestamp"] <= end_time)]

    def select_data_by_nodes_number(self, num: int = 10000):
        """
        Select data according to the top num nodes according to value
        :param num: num
        """
        # Select nodes of between EOA addresses
        eoa_and_eoa_dfm = self.dfm[(self.dfm["from_address_type"] == "EOA") & (self.dfm["to_address_type"] == "EOA")]
        eoa_and_eoa_dfm = eoa_and_eoa_dfm.sort_values("value", ascending=False).iloc[0:num, :]

        # Select nodes between Contract addresses and EOA addresses
        eoa_and_contract_dfm = self.dfm[
            ((self.dfm["from_address_type"] == "EOA") & (self.dfm["to_address_type"] == "Contract")) | (
                    (self.dfm["from_address_type"] == "Contract") &
                    self.dfm["to_address_type"] == "EOA")]
        eoa_and_contract_dfm = eoa_and_contract_dfm.sort_values("value", ascending=False).iloc[0:num, :]

        # Select nodes between contracts
        # contracts_and_contracts = self.dfm[
        #     (self.dfm["from_address_type"] == "Contract") & (self.dfm["to_address_type"] == "Contract")]
        # contracts_and_contracts = contracts_and_contracts.sort_values("value").loc[0:num, :]
        self.dfm = pd.concat([eoa_and_eoa_dfm, eoa_and_contract_dfm])
        self.dfm.sort_values("block_timestamp",inplace=True)

    def clean_and_save_data(self, start_time: datetime.datetime = datetime.datetime(2021, 9, 24, 3, 0, 0, 0, pytz.UTC),
                            end_time: datetime.datetime = datetime.datetime(2021, 9, 24, 15, 0, 0, 0, pytz.UTC),filename :str = "./data/cleaned_data.csv"):
        print("Start to add address type")
        self.add_address_type()
        print("Start to divide value by 1e18")
        self.divide_value_by_1e18()
        print("Start to select data by time")
        self.select_data_by_time(start_time, end_time)
        print("Start to select data by top active nodes")
        self.select_data_by_nodes_number()
        print("Start to save data")
        self.dfm.to_csv(filename, index=False)

    def download_data(self):
        if not os.path.exists("./data/raw_data.csv"):
            gdd.download_file_from_google_drive(file_id=self.file_id,
                                                dest_path='./data/raw_data.csv',
                                                unzip=True)
        else:
            print("File already downloaded")

    def parse_datetime(self):
        self.dfm["block_timestamp"] = self.dfm["block_timestamp"].progress_apply(lambda x: dateutil.parser.parse(x))

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
        try:
            index = self.nounce % self.w3_list_length
            res = self.w3_list[index].eth.get_code(self.w3_list[index].toChecksumAddress(address)).hex()
            self.nounce += 1
            if res == "0x":
                return "EOA"
            else:
                return "Contract"
        except:
            return "Unknown"

    def get_all_address_type(self, start_row: int, end_row: int = None) -> dict:
        """
        Get the all address type in the dataframe
        :return:  {"address1":"Contract","address2":"EOA"}
        """
        all_address = pd.DataFrame(self.dfm["from_address"].append(self.dfm["to_address"]).unique())
        all_address.rename(columns={0: "address"}, inplace=True)
        all_address["address_type"] = all_address["address"][start_row:end_row].progress_apply(self._judge_address_type)
        all_address.to_csv(f"data/all_address_{start_row}_{end_row}.csv")

    def add_address_type(self) -> None:
        """
        Define the address type as EOA(owned by a person) or Contact
        """
        # self.dfm = pd.merge(self.dfm, self.all_address, left_on='from_address', right_on='address').drop(
        #     "address").rename({"address_type": "from_address"}, inplace=True)
        # self.dfm["to_address_type"] = self.dfm["to_address"].progress_apply(self._judge_address_type)
        with open(f"data/address_type.json", "r") as f:
            address_type_dict = json.load(f)
        self.dfm.dropna(inplace=True)
        self.dfm["from_address_type"] = self.dfm["from_address"].apply(lambda x: address_type_dict[x])
        self.dfm["to_address_type"] = self.dfm["to_address"].apply(lambda x: address_type_dict[x])
        return self.dfm


def scheduler(start_row, end_row):
    preprocess_data = PreprocessData()
    preprocess_data.get_all_address_type(start_row, end_row)


class MultiProcessRequestAddressType:
    def __init__(self, process_num: int = 8, const: int = 80000):
        """
        Using multiprocess to clean the data.
        :param const: Num of data per process processes
        :param process_num: Num of process
        """
        self.const = const
        self.process_num = process_num
        self.names = locals()

    def get_all_address_type(self):
        for i in range(self.process_num):
            self.names[f"p{i}"] = mp.Process(target=scheduler, args=(i * self.const, (1 + i) * self.const))
        for i in range(self.process_num):
            self.names[f"p{i}"].start()
        for i in range(self.process_num):
            self.names[f"p{i}"].join()

    def combine_data(self):
        dfm = pd.DataFrame(columns=["address", "address_type"])
        files = os.listdir("./data")
        for i in files:
            if i.startswith("all_address_"):
                self.names[f"dfm_{i}"] = pd.read_csv(f"data/{i}", index_col=0).dropna()
                dfm = pd.concat([dfm, self.names[f"dfm_{i}"]])
                key = dfm["address"]
                value = dfm["address_type"]
                address_map = dict(zip(key, value))
                with open("data/address_type.json", "w") as f:
                    json.dump(address_map, f)
        return address_map


if __name__ == "__main__":
    # names = locals()
    # p1 = mp.Process(target=scheduler, args=(560000, 600000))
    # p2 = mp.Process(target=scheduler, args=(600000, 640000))
    # p3 = mp.Process(target=scheduler, args=(640000, 680000))
    # p4 = mp.Process(target=scheduler, args=(680000, 720000))
    # p5 = mp.Process(target=scheduler, args=(720000, 760000))
    # p6 = mp.Process(target=scheduler, args=(720000, 760000))
    # p7 = mp.Process(target=scheduler, args=(760000, 800000))
    # p8 = mp.Process(target=scheduler, args=(800000, None))
    #
    # for i in range(8):
    #     names[f"p{i + 1}"].start()
    # for i in range(5):
    #     names[f"p{i + 1}"].join()
    mp = PreprocessData()
    mp. clean_and_save_data(start_time = datetime.datetime(2021, 9, 24, 3, 0, 0, 0, pytz.UTC),
                            end_time= datetime.datetime(2021, 9, 24, 9, 0, 0, 0, pytz.UTC),filename = "./data/cleaned_data_before_ban.csv")
    mp = PreprocessData()
    mp. clean_and_save_data(start_time = datetime.datetime(2021, 9, 24, 9, 0, 0, 0, pytz.UTC),
                        end_time= datetime.datetime(2021, 9, 24, 15, 0, 0, 0, pytz.UTC),filename = "./data/cleaned_data_after_ban.csv")
                                            
