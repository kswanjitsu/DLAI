
record = ''

f = open('datasets/output/dataset_parsed_all_mimic_test_notes.txt', 'a+')

for line in open("./deidentified-medical-text-1.0/id.text", "r"):
	if line != "||||END_OF_RECORD\n":
		if "START_OF_RECORD" in line or line=='\n':
			pass
		else:	
			record = record+str(line).strip('\n')
	elif line=="||||END_OF_RECORD\n":
		record = record.strip('\n')+'\n'
		f.write(record)
		record = ''

for line in open("./deidentified-medical-text-1.0/id.res", "r"):
	if line != "||||END_OF_RECORD\n":
		if "START_OF_RECORD" in line or line=='\n':
			pass
		else:	
			record = record+str(line).strip('\n')
	elif line=="||||END_OF_RECORD\n":
		record = record.strip('\n')+'\n'
		f.write(record)
		record = ''

f.close()
