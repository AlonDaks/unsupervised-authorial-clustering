"""
This file segments the texts of Ezekiel and Jeremiah into chapters partitioned
by true authorship. We cite the following biblical studies sources to determine 
true authorship partitions.

William McKane, A Critical and Exegetical Commentary on Jeremiah (2 vols.; 
    Edinburgh: Clark, 1986-2000)

Walther Zimmerli, Ezekiel 1-2: A Commentary on the Book of the Prophet Ezekiel 
    (2 vols.; Philadelphia: Fortress, 1979-1983)

Sigmund Mowinckel, Zur Komposition des Buches Jeremia (Kristiania: Dybwad, 1914)


To Run: python segment_prophetic_strata.python
Assumes data/ezekiel_pos.txt, data/jeremiah_pos.txt exist with annotated verses
"""

import os


jeremiah_source_1 = [range(4, 19 + 1),
                     'ALL',
                     range(1, 5 + 1) + range(19, 25 + 1),
                     'ALL',
                     range(1, 17 + 1) + range(20, 31 + 1),
                     'ALL',
                     None,
                     range(4, 23 + 1),
                     range(1, 11 + 1) + range(16, 25 + 1),
                     'ALL',
                     'ALL',
                     range(1, 13 + 1),
                     range(12, 27 + 1),
                     'ALL',
                     range(5, 21 + 1),
                     'ALL',
                     range(1, 18 + 1),
                     range(13, 23 + 1),
                     None,
                     range(7, 18 + 1),
                     range(11, 14 + 1),
                     range(6, 30 + 1),
                     range(5, 29 + 1),
                     'ALL',
                     range(15, 38 + 1), ]

jeremiah_source_2 = 18 * [None, ] + [range(1, 2 + 1) + range(
    10, 11 + 1) + range(14, 15 + 1), range(1, 6 + 1)] + 5 * [None, ] + [
        'ALL', None, 'ALL', range(24, 32 + 1)
    ] + 6 * [None, ] + 8 * ['ALL', ] + [range(15, 30 + 1)]

jeremiah_source_3 = [
    None, None, range(6, 13 + 1), None, None, None, range(1, 34 + 1), range(
        1, 3 + 1), None, None, range(1, 5 + 1) + range(9, 14 + 1)
] + 6 * [None, ] + [range(1, 12 + 1), None, None, range(1, 10 + 1), range(
    1, 5 + 1)] + 2 * [None, ] + [range(1, 11 + 1), None, 'ALL', None, range(
        1, 23 + 1), None, None, range(1, 2 + 1) + range(6, 16 + 1) + range(
            24, 44 + 1), None, range(1, 7 + 1) + range(8, 22 + 1), range(
                1, 19 + 1), None, None, None, range(15, 18 + 1), None, None,
                                 None, None, range(1, 14 + 1), 'ALL']

jeremiah_source_4 = 29 * [None, ] + 2 * ['ALL']

jeremiah_source_5 = 45 * [None, ] + 6 * ['ALL']

jeremiah_source_6 = 51 * [None, ] + ['ALL']

ezekiel_source_1 = 39 * ['ALL']
ezekiel_source_2 = 39 * [None] + 9*['ALL']

def convert_prophet_book_to_dictionary(prophet_name):
	dic = {}
	f = open('data/{0}_pos.txt'.format(prophet_name))
	for line in f.readlines():
		split_line = line.split(',')
		split_line[2] = split_line[2].replace('\n', '')
		if int(split_line[0]) not in dic:
			dic[int(split_line[0])] = {int(split_line[1]) : split_line[2]}
		else:
			dic[int(split_line[0])][int(split_line[1])] = split_line[2]
	f.close()
	return dic

jeremiah = convert_prophet_book_to_dictionary('jeremiah')
ezekiel = convert_prophet_book_to_dictionary('ezekiel')

def segment_prophet(prophet_name, segment, segment_label):
	output_path = 'data/{0}_{1}_pos'.format(prophet_name, segment_label)
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	p = jeremiah if prophet_name == 'jeremiah' else ezekiel
	for i, verses in enumerate(segment):
		text = ''
		if verses == None:
			continue
		if verses == 'ALL':
			verses = p[i+1].keys()
			verses.sort()
		for verse in verses:
			text += p[i+1][verse] + 'PERIOD '
		f = open('{0}/{1}_{2}.txt'.format(output_path, prophet_name, i+1), 'w+')
		f.write(text)
		f.close()


if __name__ == '__main__':
    segment_prophet('jeremiah', jeremiah_source_1, '1')
    segment_prophet('jeremiah', jeremiah_source_2, '2')
    segment_prophet('jeremiah', jeremiah_source_3, '3')
    segment_prophet('jeremiah', jeremiah_source_4, '4')
    segment_prophet('jeremiah', jeremiah_source_5, '5')
    segment_prophet('jeremiah', jeremiah_source_6, '6')

    segment_prophet('ezekiel', ezekiel_source_1, '1')
    segment_prophet('ezekiel', ezekiel_source_2, '2')
