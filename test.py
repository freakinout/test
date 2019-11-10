from os import listdir, path
current_path = path.abspath(__file__)[:-len(path.basename(__file__))]
print(current_path)
print(path.basename(__file__))