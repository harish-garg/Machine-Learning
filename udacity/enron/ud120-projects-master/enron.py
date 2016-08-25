import re

data = [line.strip() for line in open('final_project/poi_names.txt')]
valid_data = []
for l in data:
    if re.search('\(.\)', l):
        valid_data.append(l)

print valid_data

poi_names = 0
for l in valid_data:
    if re.search('\(y\)',l):
        poi_names += 1
print "# of POI names :", len(valid_data)
