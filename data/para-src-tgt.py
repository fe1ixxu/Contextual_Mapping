import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_file", type=str, required=True)
parser.add_argument("--tgt_file", type=str, required=True)
parser.add_argument("--write_file", type=str, required=True)

args = parser.parse_args()

file1 = args.src_file #"en_token.txt"
file2 = args.tgt_file #"de_token.txt"
file3 = args.write_file #"para_en_de_small.txt"

args = parser.parse_args()

print("start parallening... at", args.write_file)
with open(file3, "w") as f3:
	with open(file1) as f1:
		with open(file2) as f2:
			line1 = f1.readline()
			line2 = f2.readline()
			while(line1):
				line = line1[:-1]+" ||| "+line2
				f3.writelines([line])
				line1 = f1.readline()
				line2 = f2.readline()


print("finished...")
