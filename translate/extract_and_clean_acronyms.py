acronym_list = []

for line in open('datasets/output/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt', 'r', encoding="utf8"):
    acronym = line.split('|')[0]
    full_wordphrase = line.split('|')[1]
    both_acronym_and_wordphrase = acronym+'|'+full_wordphrase
    both_acronym_and_wordphrase = both_acronym_and_wordphrase.strip('\n')
    if both_acronym_and_wordphrase not in acronym_list:
        acronym_list.append(both_acronym_and_wordphrase)

for line in open('./datasets/acronyms.txt', 'r', encoding="utf8"):
    acronym = line.split('|')[0]
    full_wordphrase = line.split('|')[1]
    both_acronym_and_wordphrase = acronym+'|'+full_wordphrase
    both_acronym_and_wordphrase = both_acronym_and_wordphrase.strip('\n')
    if both_acronym_and_wordphrase not in acronym_list:
        acronym_list.append(both_acronym_and_wordphrase)

for line in open('./datasets/acronyms.txt', 'r', encoding="utf8"):
    acronym = line.split('|')[0]
    both_acronym_and_wordphrase = acronym+'|'+full_wordphrase
    both_acronym_and_wordphrase = both_acronym_and_wordphrase.strip('\n')
    if both_acronym_and_wordphrase not in acronym_list:
        acronym_list.append(both_acronym_and_wordphrase)


print(len(acronym_list))

f = open('datasets/output/final_acronyms_lookup_table.txt', 'a+')
acronym_list = sorted(acronym_list)

for i in acronym_list:
    f.write(i+'\n')