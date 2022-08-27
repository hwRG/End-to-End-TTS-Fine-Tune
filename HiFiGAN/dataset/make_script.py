
a = open('transcript.txt', 'r', encoding='utf8')

lines = []
for i in range(100):
    temp = a.readline().split('|')
    lines.append(temp[0] + ' - ' + temp[1] + '\n')

a.close()

b = open('training_text.txt', 'w', encoding='utf8')

for line in lines:
    b.write(line)
b.close()