import os

path_train_polar_labels = '/home/rblin/Images/BD_ITSC/train_polar/LABELS/'
path_train_polar = '/home/rblin/Images/BD_ITSC/train_polar/PARAM_POLAR/'

list_train_polar_labels = os.listdir(path_train_polar_labels)

for i in list_train_polar_labels:
    f = open(path_train_polar_labels + i, 'r')
    content = f.read()
    if 'car' not in content and 'bike' not in content and 'motorbike' not in content and 'person' not in content:
        name = i.split('.')
        if os.path.exists(path_train_polar + 'RetinaNet_I/' + name[0] + '.png'):
            os.remove(path_train_polar + 'RetinaNet_I/' + name[0] + '.png')
        if os.path.exists(path_train_polar + 'RetinaNet_Params/' + name[0] + '.png'):
            os.remove(path_train_polar + 'RetinaNet_Params/' + name[0] + '.png')
        if os.path.exists(path_train_polar + 'RetinaNet_Stokes/' + name[0] + '.png'):
            os.remove(path_train_polar + 'RetinaNet_Stokes/' + name[0] + '.png')
        if os.path.exists(path_train_polar_labels + i):
            os.remove(path_train_polar_labels + i)
    f.close()
