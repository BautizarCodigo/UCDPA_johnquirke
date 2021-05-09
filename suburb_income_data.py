from bs4 import BeautifulSoup
from urllib.request import urlopen
import pprint

class PerthIncomeSuburb:

    def fetch_data(self):
        ''' Fetches the income data from the suburbs of perth'''

        html_doc = 'http://house.speakingsame.com/suburbtop.php?sta=wa&cat=Median+household+income&name=Weekly+income'
        soup = BeautifulSoup(urlopen(html_doc), "html.parser")
        #text = soup.get_text()

        #print(text)

        #Get the Table data
        table = soup.find(lambda tag: tag.name == 'table' and tag.has_attr('id') and tag['id'] == "mainT")
        rows = table.findAll(lambda tag: tag.name == 'tr')[42:92]

        incomes = []
        for i in rows:
            incomes.append(i.get_text())


        for details in incomes:
            print(details)
            # i.encode("ascii", "ignore")
            # print(i.split("$"))
















if __name__ == "__main__":
    income = PerthIncomeSuburb()
    income.fetch_data()


