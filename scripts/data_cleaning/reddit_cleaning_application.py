'''
Reddit Data Cleaning Application
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import emoji
import shutil
import stat
import unidecode

# import specific functions
from reddit_cleaning_functions import *

'''
Part 1: General Cleaning
'''

# import data - student loan forgivness
reddit_v1_1 = pd.read_csv('../data_acquisition/reddit_data/student_loan_forgiveness_1_18_25.csv')
reddit_v1_2 = pd.read_csv('../data_acquisition/reddit_data/student_loan_forgiveness_1_20_25.csv')
# import data - student loans
reddit_v2_1 = pd.read_csv('../data_acquisition/reddit_data/student_loans_1_20_25.csv')
# import data - is a college degree worth it
reddit_v3_1 = pd.read_csv('../data_acquisition/reddit_data/degree_worth_1_20_25.csv')

# concatenate v1 data
reddit_v1 = pd.concat([reddit_v1_1, reddit_v1_2], ignore_index=True)

# check for duplicates
reddit_v1.drop_duplicates(inplace=True, ignore_index=True)
reddit_v2_1.drop_duplicates(inplace=True, ignore_index=True)
reddit_v3_1.drop_duplicates(inplace=True, ignore_index=True)

# clean data
reddit_v1_clean = clean_reddit_extraction(reddit_v1)
reddit_v2_clean = clean_reddit_extraction(reddit_v2_1)
reddit_v3_clean = clean_reddit_extraction(reddit_v3_1)

# save data - important for utf-8 encoding
reddit_v1_clean.to_csv('reddit_data_cleaned/student_loan_forgiveness.csv', encoding='utf-8', index=False)
reddit_v2_clean.to_csv('reddit_data_cleaned/student_loans.csv', encoding='utf-8', index=False)
reddit_v3_clean.to_csv('reddit_data_cleaned/degree_worth.csv', encoding='utf-8', index=False)

# import cleaned data when required
student_loan_forgiveness = pd.read_csv('reddit_data_cleaned/student_loan_forgiveness.csv')
student_loans = pd.read_csv('reddit_data_cleaned/student_loans.csv')
degree_worth = pd.read_csv('reddit_data_cleaned/degree_worth.csv')

'''
Part 2: Labeling and User Mapping
'''
# apply organizing function
labeled_student_loan_forgiveness = organize_reddit_content(student_loan_forgiveness)
labeled_student_loans = organize_reddit_content(student_loans)
labeled_degree_worth = organize_reddit_content(degree_worth)

# save data
labeled_student_loan_forgiveness.to_csv('reddit_data_cleaned/labeled_student_loan_forgiveness.csv', encoding='utf-8', index=False)
labeled_student_loans.to_csv('reddit_data_cleaned/labeled_student_loans.csv', encoding='utf-8', index=False)
labeled_degree_worth.to_csv('reddit_data_cleaned/labeled_degree_worth.csv', encoding='utf-8', index=False)

'''
# create corpora - opting for content version
create_reddit_corpus('student_loan_forgiveness', student_loan_forgiveness, location='reddit_data_cleaned/corpora', replace_folder=True)
create_reddit_corpus('student_loans', student_loans, location='reddit_data_cleaned/corpora', replace_folder=True)
create_reddit_corpus('degree_worth', degree_worth, location='reddit_data_cleaned/corpora', replace_folder=True)
'''
