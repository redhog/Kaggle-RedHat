import csv
import re
import datetime
import numpy
import sys

def csv2dict(fileName):
    rows = []
    with open (fileName + '.csv') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in row.keys():
            if re.search(r'char_', col):
                if (row[col] != ''):
                    row[col] = float(row[col].split(' ')[1])
                else:
                    row[col] = 0
            elif (col == 'people_id'):
                row['people_id'] = float(row['people_id'].split('_')[1])
            elif (col == 'activity_id'):
                row['activity_id'] = float((row['activity_id'].split('_')[1])) + 0.1 * float(re.search(r'(\d{1})', row['activity_id'].split('_')[0]).groups()[0]) #fuck it this MUST be easier!
            elif (col == 'date'):
                row['date'] = (datetime.datetime.strptime(row['date'], "%Y-%m-%d") - datetime.datetime(1970, 1, 1)).total_seconds()
            
        if row['activity_category']:
            row['activity_category'] = row['activity_category'].split(' ')[1]
    return rows

def dict2npz(rows, fileName):
    arr = numpy.zeros(len(rows), dtype=[(key, "f4") for key in rows[0].iterkeys()])
    for idx, row in enumerate(rows):
        for key, value in row.iteritems():
            arr[idx][key] = value
    numpy.savez_compressed(fileName + ".npz", x=arr)
    



fileName = sys.argv[1]
dict2npz(csv2dict(fileName), fileName)
