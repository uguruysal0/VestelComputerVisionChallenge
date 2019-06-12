"""
Bu script oluturualn submission kaggle için gerekli formata getiriyor.
"""

import csv

class_to_index = {
    "bleach_with_non_chlorine":0,
    "do_not_bleach":1,
    "do_not_dryclean":2,
    "do_not_tumble_dry":3,
    "do_not_wash":4,
    "double_bar":5,
    "dryclean":6,
    "low_temperature_tumble_dry":7,
    "normal_temperature_tumble_dry":8,
    "single_bar":9,
    "tumble_dry":10,
    "wash_30":11,
    "wash_40":12,
    "wash_60":13,
    "wash_hand":14,
}
index_to_class = { class_to_index[i]:i for i in class_to_index }

base = "test_"
FILENAME =  "output.csv"
counter = 1
predictions = {}
with open(FILENAME,"r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    c = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row[1] not in predictions:
                predictions[row[1]] = []
                predictions[row[1]].append(row)
            else:
                predictions[row[1]].append(row)
result = []
ctr = 1

while (base+str(counter)) in predictions:
    for item in predictions[base+str(counter)]:
        item[0] = ctr
        ctr+=1
        result.append(item)
    counter +=1    

with open("ready_submission.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerows(result)