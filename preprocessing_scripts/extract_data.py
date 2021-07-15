# TODO: NOT REMOVE PART OF FRAMEWORK!!!!!!!!!!!!!!!!
#  replace after developing

import re
import os
from glob import glob

import argparse
import traceback

import pandas as pd
from tqdm import tqdm
from colorama import Fore

def extract_info(input_path, output_path, columns) -> bool:
    """
    Args:
        input_path (str): The first parameter, path of one file.
        output_path (str): The second parameter, path of directory.

    Returns:
        bool: The return value. True for success, False otherwise.

    """

    df = pd.read_table(input_path, usecols=columns)

    # filter data: drop rows which contain any missing values.
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    try:
        switch = {'-': 'negative', '+': 'positive'}
        # extract target values if not containe pass to another patient
        target = re.search(r'(?<=Cytomegalovirus\s).', df.sample_tags[0])[0]
        df['target'] = target

        df['target'] = df['target'].map(switch)

        # filter data: drop all sequences with
        mask = df.amino_acid.str.contains('*', regex=False)
        df = df[~mask]
        df.reset_index(drop=True, inplace=True)

        cols = ['v_family', 'amino_acid', 'j_family']
        df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        df.drop(['sample_tags', 'v_family', 'amino_acid', 'j_family'], axis=1, inplace=True)

        new_path = ''.join([output_path, df.sample_name[0], '_', switch[target], '.csv'])

        df.to_csv(new_path, index=False)
        return True

    except Exception as e:
        print(f'There is no target in sample_tags columns {e}', end='\n' * 2)
	return None	

def main(inputdir, outputdir):
    files = glob(inputdir + '*')

    columns = ['sample_tags', 'v_family', 'j_family', 'amino_acid', 'sample_name']
    successed = [extract_info(file, outputdir, columns) for file in tqdm(files, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET))]

    print('Successfully convert and extract from: {} files'.format(sum(successed)))


if __name__ == '__main__':
    try:

        train_output = os.path.join(os.getcwd(), 'Data_New', 'PreprocessedTrain/')
        if not os.path.exists(train_output):
            os.makedirs(train_output)

        test_output = os.path.join(os.getcwd(), 'Data_New', 'PreprocessedTest/')
        if not os.path.exists(test_output):
            os.makedirs(test_output)

        parser = argparse.ArgumentParser(
            description='python extract_data.py --train_inputdir=Data/Train --train_outputdir=Data/PreprocessedTrain \
             --test_inputdir=Data/Test --test_outputdir=Data/PreprocessedTest')

        parser.add_argument('--train_inputdir', type=str, default='Data/Train/', nargs='?', help='Input dir of train files')

        parser.add_argument('--train_outputdir', type=str, default=train_output, nargs='?',
                            help='Output dir for train files')

        parser.add_argument('--test_inputdir', type=str, default='Data/Test/', nargs='?',
                            help='Input dir of test files')

        parser.add_argument('--test_outputdir', type=str, default=test_output, nargs='?',
                            help='Output dir for test files')

        args = parser.parse_args()

        # main(args.train_inputdir, args.train_outputdir)
        main(args.test_inputdir, args.test_outputdir)

    except Exception as e:
        print(e)
        print(traceback.format_exc())
