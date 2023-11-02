import csv

with open('/home/vant/code/tfm1/data/mindboggle_ROIs.txt', 'r') as txtfile, open('/home/vant/code/tfm1/data/nodes.csv', 'w', newline='') as csvfile:
    reader = csv.reader(txtfile, delimiter='\t')
    writer = csv.writer(csvfile, delimiter=',')
    for row in reader:
        writer.writerow(row)

