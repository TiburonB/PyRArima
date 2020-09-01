This directory contains all you should need to prepare a sample submission for spatio-temporal challenges.

The code was tested with:
Anaconda Python 3.7 

Usage:

(1) If you are a challenge participant:

- The files sample_code_submission.zip contains a sample submission ready to go!

- The file README.ipynb contains step-by-step instructions on how to create a sample submission. 
At the prompt type:
jupyter-notebook README.ipynb

Without running the jupyter-notebook, you can also directly:

- Modify sample_code_submission/model.py to provide a better model and win the challenge!

- Check your code in the same conditions it will be run on the platform, using the command:

`python ingestion_program/ingestion.py sample_data sample_results ingestion_program sample_code_submission`

- Download a larger dataset (called public_data) from the website of the challenge and re-test your code by replacing sample_data by public_data.

- To create a submission, zip the contents of sample_code_submission (without the directory, but with metadata)


(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)

- the following programs run properly:

    `python ingestion_program/ingestion.py sample_data sample_results ingestion_program sample_code_submission`

    `python scoring_program/score.py sample_data sample_results scoring_output`

- the metric identified in metric.py is the metric used both to compute performances in README.ipynb and for the challenge.

- your code also runs within the Codalab docker (inside the docker, python 3.6 is called python3):

	`docker run -it -v `pwd`:/home/aux codalab/codalab-legacy:py3`
	
	`DockerPrompt# cd /home/aux`
	`DockerPrompt# python3 ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`
	`DockerPrompt# python3 scoring_program/score.py sample_data sample_result_submission scoring_output`
	`DockerPrompt# exit`