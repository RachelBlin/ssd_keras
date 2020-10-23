import os

path_train_polar = '/home/rblin/Images/BD_ITSC/train_polar/PARAM_POLAR/RetinaNet_I'
path_polar_train_txt = '/home/rblin/Images/BD_ITSC/train_polar/PARAM_POLAR/train_polar.txt'

path_test_polar = '/home/rblin/Images/BD_ITSC/test_polar/PARAM_POLAR/RetinaNet_I'
path_polar_test_txt = '/home/rblin/Images/BD_ITSC/test_polar/PARAM_POLAR/test_polar.txt'

path_test_rgb = '/home/rblin/Images/BD_ITSC/test_rgb/RGB'
path_rgb_test_txt = '/home/rblin/Images/BD_ITSC/test_rgb/test_rgb.txt'

path_test_rgb = '/home/rblin/Images/BD_ITSC/test_rgb/RGB'
path_rgb_test_txt = '/home/rblin/Images/BD_ITSC/test_rgb/test_rgb.txt'

list_train_polar = os.listdir(path_train_polar)

list_test_polar = os.listdir(path_test_polar)

list_test_rgb = os.listdir(path_test_rgb)

f = open(path_polar_train_txt, 'w')
for i in list_train_polar:
    name = i.split('.')
    f.write(name[0] + "\n")

f.close()

f = open(path_polar_test_txt, 'w')
for i in list_test_polar:
    name = i.split('.')
    f.write(name[0] + "\n")

f.close()

f = open(path_rgb_test_txt, 'w')
for i in list_test_rgb:
    name = i.split('.')
    f.write(name[0] + "\n")

f.close()

