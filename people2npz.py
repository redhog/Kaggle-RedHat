import csv
import datetime
import numpy

def convert(row):
    for col in row:
        if col == 'date':
            row['date'] = (datetime.datetime.strptime(row['date'], "%Y-%m-%d") - datetime.datetime(1970, 1, 1)).total_seconds()
        else:
            if row[col] == 'False':
                row[col] = 0.0
            elif row[col] == 'True':
                row[col] = 1.0
            else:
                try:
                    row[col] = row[col].replace("_", " ")
                    if ' ' in row[col]: row[col] = row[col].split(' ')[1]
                    row[col] = float(row[col])
                except Exception, e:
                    print e
                    import sys, pdb
                    sys.last_traceback = sys.exc_info()[2]
                    pdb.pm()
    return row

with open("people.csv") as f:
    rows = [convert(row) for row in csv.DictReader(f)]
        
arr = numpy.zeros(len(rows), dtype=[(key, "f4") for key in rows[0].iterkeys()])
for idx, row in enumerate(rows):
    for key, value in row.iteritems():
        arr[idx][key] = value

numpy.savez_compressed("people.npz", x=arr)

    
