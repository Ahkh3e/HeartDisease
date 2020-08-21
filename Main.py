from RFCmodel import RFC
from DTCmodel import DTC
from Data import graphs
import pandas as pd
data = pd.read_csv('train.csv')

graphs(data)

RFC(data)

DTC(data)
