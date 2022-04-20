file = ['glove.twitter.27B.25d.txt','glove.twitter.27B.50d.txt','glove.twitter.27B.100d.txt','glove.twitter.27B.200d.txt']
vectorSize = 0
wordNum = 0
for name in file:
    with open(name, 'r') as f:
        line = f.readline().split(' ')
        vectorSize = len(line) - 1
        
    for wordNum, line in enumerate(open(name, 'r')):
        pass
    print('wordNum: {:d}, vectorSize: {:d}'.format(wordNum, vectorSize))
    with open(name, 'r') as f:
        content = f.readlines()
    with open(name[0:-4] + '.gensim.txt', 'w') as f:
        f.write('{:d} {:d}\n'.format(wordNum, vectorSize))
        for line in content[0 : len(content) // 2]:
            f.write(line)
    with open(name[0:-4] + '.gensim.txt', 'a') as f:
        for line in content[len(content) // 2:]:
            f.write(line)
    