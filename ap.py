recoms = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]  # N = 10
precs = []

for indx, rec in enumerate(recoms):
    precs.append(sum(recoms[:indx+1])/(indx+1))

print(precs)
