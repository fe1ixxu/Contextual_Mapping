import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--en_file", type=str, required=True)
parser.add_argument("--lg_file", type=str, required=True)

args = parser.parse_args()

file_name = args.file  #"en-de.txt"
write_name = args.en_file #"./en.txt"
write_name2 = args.lg_file #"./de.txt"

print("start creating sentences....")

i=0
pre_word = 0
with open(file_name) as f:
	with open(write_name, "w") as f2:
		with open(write_name2, "w") as f3:
			line = f.readline()

			while(line and i<1000000):
				L = line.split('\t')
				length = len(L[0])
				if pre_word != length and len(L[0].split())<150:
					f2.writelines([L[0],"\n"])
					f3.writelines(L[-1])
					pre_word = length
					line = f.readline()
					i += 1
				else:
					line = f.readline()

print("finished....")