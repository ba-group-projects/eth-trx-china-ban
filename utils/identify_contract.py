import requests
from bs4 import BeautifulSoup


def identify_contract(contract_address):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
    }
    url = "https://etherscan.io/address/" + contract_address
    response = requests.get(url,headers=headers).text
    soup = BeautifulSoup(response, 'html.parser')
    title =soup.find('title').text.strip().split("|")[0]
    return title


if __name__ == '__main__':
    identify_contract('0xE592427A0AEce92De3Edee1F18E0157C05861564')
