import sys
from random import randint
print 

file_name = sys.argv[1]
all_classes = dict()

with open(file_name, 'r') as f:
  for line in f:
    splitted = line.strip().split(' ')
    label = splitted[1]
    if label is all_classes.keys():
      all_classes[label].append(line);
    else:
      all_classes[label] = list()
      all_classes[label].append(line);
      
#choose one image from each all_class
list_images = list()
for key,elem in all_classes.iteritems():
    num_elem  = len(elem)
    rand_elem =   randint(0,num_elem-1)
    list_images.append(elem[rand_elem])
    
#save file
with open(sys.argv[2], 'wb') as f:
  f.writelines( "%s" % item for item in list_images )
  
