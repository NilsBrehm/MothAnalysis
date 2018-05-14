import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import myfunctions as mf
import csv
from IPython import embed

source = 'Food and Agriculture Organization of the United Nations'

year = [1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,\
       1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,\
       2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]

hives = [1997000,1761000,1743000,1689600,1720600,1758900,1650700,1562700,1612442,1496784,1484901,1486914,1452825,
         1326174,1588177,1489471,1506747,1580848,1533953,1528800,1550800,1563100,1544600,1671000,1677100,1761600,
         1886200,1818700,1805900,1605000,1215000,1168000,1120000,1085000,1047000,1028000,952000,917000,899000,902000,
         892723,881987,838832,750525,738933,698290,669999,691689,694851,685441,695052,698714,708775,741461,771851,
         742968]

world = [49173473,49639027,49970283,49625534,50368109,52083386,52288194,52573800,53478677,54019911,54174200,54482023,\
        55310535,55195675,55851124,55740283,56204567,57706024,58491794,60198428,62248861,63622700,64671383,66060290,\
        67458523,67129592,67443256,68277761,69236607,69237913,69960216,67499709,66712015,66110859,66171863,66000573,\
        65941668,66613419,67075297,69300827,70393665,71828958,72009099,72973177,74276465,75516681,74967203,76119487,\
        77095060,79683535,80403617,83059261,84899038,87262892,89011674,90564654]

# with open('/home/brehm/Downloads/beehives.csv', 'r') as csvfile:
#     embed()

mf.plot_settings()
fig, ax1 = plt.subplots()

ax1.plot(year, np.array(hives)/10**6, 'ko-', label='Germany')
ax2 = ax1.twinx()
ax2.plot(year, np.array(world)/10**6, 'ro-', label='Worldwide')
plt.xlabel('Year')
ax1.set_ylabel('Bee Hives [in millions]')
ax2.set_ylabel('Bee Hives [in millions]', color='r')
plt.title('Number of Bee Hives ')
ax1.text(1960, 0.7, source, size=8)
ax2.set_xticks(np.arange(1960, 2020, 5))
# sns.despine()
fig.legend()
plt.show()