import glob

fitpaths = glob.glob('matilija.calibrate.batch*/particle*/fitness.log')
for fp in fitpaths:
    print(fp)
    with open(fp, 'r') as ff:
        lines = ff.readlines()
    #for i, line in enumerate(lines):
    #    if i == 1 and line.startswith('score'):
    #        print(line)
    with open(fp, 'w') as ffw:
        for i, line in enumerate(lines):
            if i == 1 and line.startswith('score'):
                continue
            ffw.write(line)
