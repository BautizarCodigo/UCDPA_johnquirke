import csv
import re


from bs4 import BeautifulSoup
from urllib.request import urlopen
import pprint

class PerthIncomeSuburb:

    def fetch_data(self):
        """Fetches the income data from the suburbs of perth"""

        base_url = 'http://house.speakingsame.com/suburbtop.php?sta=wa&cat=Median+household+income&name=Weekly+income&page='
        pages= [0, 1, 2, 3]
        locations = []
        income = []

        for page in range(len(pages)):
            soup = BeautifulSoup(urlopen(base_url + str(page ) ), "html.parser")
            table = soup.find(lambda tag: tag.name == 'table' and tag.has_attr('id') and tag['id'] == "mainT")
            rows = table.findAll(lambda tag: tag.name == 'tr')[42:92]

            for i in rows:
                #Add Suburbs
                locations.append(i.get_text(" ")[3:-7].strip())

                # Low income Suburbs
                if '$' in i.get_text(" ")[-4:]:
                    income.append(i.get_text(" ")[-3:].replace(',', ''))
                else:
                    income.append(i.get_text(" ")[-5:].replace(',', ''))

        weekly_incomes = list(zip(locations,income))

        with open('suburb_Weekly_income.csv', 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Suburb','Weekly Income'])

            for row in weekly_incomes[:-1]: #Removes << Page, Next page >>
                writer.writerow(row)



if __name__ == "__main__":
    income = PerthIncomeSuburb()
    income.fetch_data()


