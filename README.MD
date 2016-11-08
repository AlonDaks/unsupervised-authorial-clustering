### Code and Data Repository

Alon Daks and Aidan Clark. “Unsupervised Authorial Clustering Based on Syntactic Structure”. *ACL 2016*, 114. 
https://www.aclweb.org/anthology/P/P16/P16-3017.pdf

### Installation ###

This project requires python 2.7.9 along with several packages. One can install
the necessary packages using pip, by doing `pip install -r requirements.txt`.

N.B. The Natural Language Toolkit (NLTK) is a large installation that is not 
required for running our code on the data we have supplied, but will be required
to test our framework on new datasets.

**Data**

Unzip data.zip to a directory named 'data' residing in the repository's 
root directory. 

MD5 checksum for data.zip: `811eac7b78d453160bc231073174ea57`

### Reproducing Our Results ###

To run all analyses, do `make all` from the project's root directory. 

To run analysis by dataset do `make nyt`, `make sanditon`, `make campaign`, or
`make bible` to run the new york times, sanditon, 2008 campaign speeches, or 
biblical datasets respectively. 

### Source Code ###

- `cluster.py` 
	The primary source code of our submission. cluster.py contains all primary 
	code to run authorial decomposition as described in our paper. Details on 
	how to run this file can be found above, in the secton titled, 
	"Installation".
- `util.py`
	A short collection of helper functions defined for use in cluster.py. 
- `convert_to_pos.py`
	A collection of functions designed to translate a plain .txt document 
	written in English to POS tokens, as manufactured by the Natural Language 
	Toolkit, as described in our paper. 	
	Documentation within this file describes how to use it to covert documents.
- `segment_prophetic_strata.py`
	A collection of code which partions the Book of Ezekiel into the two authors
	which produced the Book of Ezekiel. Also partions the Book of Jeremiah into
	the six authors that produced the Book of Jeremiah. In both cases, it does
	so based on the Biblical Scholarship cited in documentation within this 
	file.

### Directory Structure ###

All source code is contained in the root. All data for use in testing is 
contained within the 'data' directory. Within the 'data' directory is a number 
of subdirectories, each containing the work of a single author in the form of a 
several .txt files, where each .txt file is considered by our framework to be a 
'document'. 

As an example, the directory `./data/obama/` contains a list of 27 .txt files, 
each containing the transcript of a single speech from President Obama's 2008 
campaign.

Several filename tags are used to denote specific information about the files.

- `_X` appended onto the end of a directory name, where X is any integer, 
  signifies that it contains documents from the Xth sub-author of a larger
  work ostensibly by one author. As an example, the directory 
  `./data/ezekiel_2/` contains all chapters in the Book of Ezekiel 
  believed to be written by the author we call Ezekiel 2.
- `_english` appended onto the end of a directory name signifies that it
  contains the documents translated from Hebrew into English. This is only
  used on texts from the Hebrew Bible.
- `_pos` appended onto the end of a directory name signifies that it 
  contains the part-of-speech converted documents rather than the orignal 
  documents. 

A directory must have `_pos` appended onto its name if it is going to be used in
our clustering framework with part-of-speech n-grams. Furthermore, if a 
directory has a combination of the suffixes listed above, they must appear in 
the order they have just been given.

There are additionally two files, `./data/ezekiel_pos.txt` and 
`./data/jeremiah_pos.txt`, which contain part-of-speech data for the Books of 
Ezekiel and Jeremiah for use in segment_prophetic_strata.py.

An explanation for all subdirectories within `./data/` now follows.

**New York Times Columnists**

- `./data/collins_pos/`
  274 columns from NYT columnist Gail Collins, having been converted to 
  part-of-speech representations as outlined in our paper.
- `./data/dowd_pos/`
  298 columns from NYT columnist Maureen Dowd, having been converted to 
  part-of-speech representations as outlined in our paper.
- `./data/friedman_pos/`
  279 columns from NYT columnist Thomas Friedman, having been converted to
  part-of-speech representations as outlined in our paper.
- `./data/krugman_pos`
  331 columns from NYT columnist Paul Krugman, having been converted to 
  part-of-speech representations as outlined in our paper.

**The Books of Ezekiel and Jeremiah**

(English text obtained from Project Gutenburg, Hebrew text obtained from 
LAF-Fabric: http://laf-fabric.readthedocs.org/en/latest/index.html)

- `./data/ezekiel_1/`
  The 39 chapters of the Hebrew text of the Book of Ezekiel attributed to 
  Ezekiel 1.
- `./data/ezekiel_1_pos/`
  The 39 chapters of the Hebrew text of the Book of Ezekiel attributed to 
  Ezekiel 1, having been converted to part-of-speech representations as 
  outlined in our paper.
- `./data/ezekiel_2/`
  The 9 chapters of the Hebrew text of the Book of Ezekiel attributed to 
  Ezekiel 2.
- `./data/ezekiel_2_pos/`
  The 9 chapters of the Hebrew text of the Book of Ezekiel attributed to 
  Ezekiel 2, having been converted to part-of-speech representations as 
  outlined in our paper.
- `./data/ezekiel_english/`
  All 48 chapters of the English King James Version of the Book of 
  Ezekiel.
- `./data/ezekiel_english_pos/`
  All 48 chapters of the English King James Version of the Book of 
  Ezekiel, having been converted to part-of-speech representations as 
  outlined in our paper.
- `./data/ezekiel_pos/`
  All 48 chapters of the Hebrew text of the Book of Ezekiel, having been 
  converted to part-of-speech representations as outlined in our paper.
- `./data/jeremiah_1/`
  The 23 chapters of the Hebrew text of the Book of Jeremiah attributed to
  Jeremiah 1.
- `./data/jeremiah_1_pos/`
  The 23 chapters of the Hebrew text of the Book of Jeremiah attributed to
  Jeremiah 1, having been converted to part-of-speech representations as 
  outlined in our paper.
- `./data/jeremiah_2/`
  The 14 chapters of the Hebrew text of the Book of Jeremiah attributed to
  Jeremiah 2.
- `./data/jeremiah_2_pos/`
  The 14 chapters of the Hebrew text of the Book of Jeremiah attributed to
  Jeremiah 2, having been converted to part-of-speech representations as 
  outlined in our paper.
- `./data/jeremiah_english/`
  All 52 chapters of the English King James Version of the Book of 
  Jeremiah.
- `./data/jeremiah_english_pos/`
  All 52 chapters of the English King James Version of the Book of 
  Jeremiah, having been converted to part-of-speech representations as 
  outlined in our paper.

**The Novel Sanditon**

(Obtained from http://etext.lib.virginia.edu/toc/modeng/public/AusSndt.html)

- `./data/austen/`
  11 chapters of the novel Sanditon as written by Jane Austen.
- `./data/austen_pos/`
  11 chapters of the novel Sanditon as written by Jane Austen, having been
  converted to part-of-speech representations as outlined in our paper.
- `./data/lady/`
  19 chapters of the novel Sanditon as completed by an Anonymous Woman.
- `./data/lady_pos/`
  19 chapters of the novel Sanditon as completed by an Anonymous Woman, 
  having been converted to part-of-speech representations as outlined in 
  our paper.

**Election Speeches from 2008**

(Obtained from http://www.presidentialrhetoric.com/campaign2008/)

- `./data/mccain/`
  20 speeches given by Senator McCain during the 2008 election season.
- `./data/mccain_pos/`
  20 speeches given by Senator McCain during the 2008 election season, 
  having been converted to part-of-speech representations as outlined in 
  our paper.
- `./data/obama/`
  27 speeches given by President Obama during the 2008 election season.
- `./data/obama_pos/`
  27 speeches given by President Obama during the 2008 election season, 
  having been converted to part-of-speech representations as outlined in 
  our paper.


N.B. Not all text-file directories exist for `./data/*_pos/` part-of-speech 
converted directories.

### Citation ###
If you use our framework or code in a publication, we would appreciate citations.
```
@article{daks2016unsupervised,
  title={Unsupervised Authorial Clustering Based on Syntactic Structure},
  author={Daks, Alon and Clark, Aidan},
  journal={ACL 2016},
  pages={114},
  year={2016}
}
```
