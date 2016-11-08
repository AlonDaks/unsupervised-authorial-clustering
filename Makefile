.PHONY: all nyt sanditon campaign bible nyt-validation

all:
	make nyt
	make nyt-validation
	make sanditon
	make campaign
	make bible

nyt:
	python cluster.py --authors='dowd, krugman' --pos=True --k=2 \
		--randomize=True --num_runs=10
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=10
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=3 --randomize=True --num_runs=10

nyt-validation:
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(1,1)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(2,2)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(3,3)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(4,4)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(5,5)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(6,6)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(1,2)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(1,3)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(1,4)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(1,5)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(1,6)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(2,3)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(2,4)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(2,5)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(2,6)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(3,4)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(3,5)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(3,6)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(4,5)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(4,6)'
	python cluster.py --authors='friedman, krugman, collins, dowd' --pos=True \
		--k=2 --randomize=True --num_runs=2 --ngram_range='(5,6)'

sanditon:
	python cluster.py --authors='austen, lady' --pos=True --k=2 \
		 --randomize=True --num_runs=10

campaign:
	python cluster.py --authors='obama, mccain' --pos=True --k=2 \
		 --randomize=True --num_runs=10
	python cluster.py --authors='obama, mccain' --pos=False --k=2 \
		 --randomize=True --num_runs=10 --ngram_range='(1,1)'

bible:
	python cluster.py --authors='ezekiel_english, jeremiah_english' --pos=True \
		--k=2 --randomize=True --num_runs=10
	python cluster.py --authors='ezekiel_english, jeremiah_english' \
		--pos=False --k=2 --randomize=True --num_runs=10

	python cluster.py --authors='ezekiel_1,ezekiel_2' --pos=True --k=2 \
		 --randomize=True --num_runs=10
	python cluster.py --authors='jeremiah_1,jeremiah_2' --pos=True --k=2 \
		 --randomize=True --num_runs=10
	python cluster.py --authors='ezekiel_1,ezekiel_2,jeremiah_1,jeremiah_2' \
					--pos=True --k=4 --randomize=True --num_runs=10

	python cluster.py --authors='ezekiel_1,ezekiel_2' --pos=False --k=2 \
		 --randomize=True --num_runs=10
	python cluster.py --authors='jeremiah_1,jeremiah_2' --pos=False --k=2 \
		 --randomize=True --num_runs=10
	python cluster.py --authors='ezekiel_1,ezekiel_2,jeremiah_1,jeremiah_2' \
					--pos=False --k=4 --randomize=True --num_runs=10
