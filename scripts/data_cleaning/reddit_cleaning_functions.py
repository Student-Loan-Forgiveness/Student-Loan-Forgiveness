'''
Reddit Data Cleaning Functions
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import emoji # emoji.replace_emoji(text, replace='')
import shutil
import stat
import unidecode

'''
Cleaning Process for Content:
    0. If content is posting from outside source:
        - content will be float type
        - replace with title
    1. remove '>'
    2. remove '\n'
    3. remove '\xa0'
    4. remove links and email addresses
    5. remove emojis and other non-ASCII characters
    6. remove multispaces
    7. strip()
'''

# function remove website links
def remove_links(string):
    result = re.sub(r'http\S+|www\.\S+', '', string)
    return result

# function to remove email addresses
def remove_email_addresses(string):
    result = re.sub(r'\S+@\S+', '', string)
    return result.strip()

# function to remove multiple spaces
def remove_multiple_spaces(string):
    return re.sub(r'\s+', ' ', string).strip()

# function to remove emojis
def remove_emojis(string):
    return emoji.replace_emoji(string, replace='')

# function to replace empty content root posts with title
def replace_empty_root(row):
    if (row['replying_to'] == 'Root') and (len(row['content']) == 0):
        return row['title']
    else:
        return row['content']

# function to clean reddit extraction
def clean_reddit_extraction(reddit_extraction):
    # create copy
    cleaned_extraction = reddit_extraction.copy()
    
    # clean content and title
    clean_sections = ['content', 'title']
    
    for section in clean_sections:
        # outside content check
        cleaned_extraction[section] = cleaned_extraction[section].fillna('')
        
        # remove '>'
        cleaned_extraction[section] = cleaned_extraction[section].str.replace('>', '')
        
        # remove '\n'
        cleaned_extraction[section] = cleaned_extraction[section].str.replace('\n', '')
        
        # remove '\xa0'
        cleaned_extraction[section] = cleaned_extraction[section].str.replace('\xa0', '')
        
        # remove links
        cleaned_extraction[section] = cleaned_extraction[section].apply(lambda row: remove_links(row))
        
        # remove email addresses
        cleaned_extraction[section] = cleaned_extraction[section].apply(lambda row: remove_email_addresses(row))
        
        # remove emojis
        cleaned_extraction[section] = cleaned_extraction[section].apply(lambda row: remove_emojis(row))
        
        # remove non-ASCII characters
        cleaned_extraction[section] = cleaned_extraction[section].apply(lambda row: unidecode.unidecode(row))
        
        # remove multispaces
        cleaned_extraction[section] = cleaned_extraction[section].apply(lambda row: remove_multiple_spaces(row))
        
        # strip
        cleaned_extraction[section] = cleaned_extraction[section].str.strip()
    
    # if content is root post and length is 0 then fill with title
    cleaned_extraction['content'] = cleaned_extraction.apply(replace_empty_root, axis=1)
    
    return cleaned_extraction

# function to create author mapping
def create_author_map(reddit_post):
    # data structure
    author_map = {'author': [],
                  'submissions': [],
                  'replies_to': []}
    
    # get authors
    authors = reddit_post['author'].unique().tolist()
    
    # replies_from dictionary
    replies_from_dict = {author:[] for author in authors}
    
    # iterate through authors to get associated submissions (replies_to)
    for author in authors:
        author_submissions = reddit_post[reddit_post['author']==author]['id'].tolist()
        author_replies = reddit_post[reddit_post['author']==author]['replying_to'].tolist()
        
        # initialize replies_to
        replies_to = []
        
        # iterate through replies
        for reply in author_replies:
            # ignore initial submission
            if reply != 'Root':
                try:
                    # author to
                    author_to = reddit_post[reddit_post['id']==reply.split('_')[1]]['author'].values[0]
                    replies_to.append(author_to)
                    
                    # author from
                    replies_from_dict[author_to].append(author)
                    
                except IndexError:
                    continue
                
        # populate data structure
        author_map['author'].append(author)
        author_map['submissions'].append(author_submissions)
        author_map['replies_to'].append(replies_to)
        
    # convert author_map to pandas dataframe
    author_map_df = pd.DataFrame(author_map)
    
    # make new column for replies_from
    author_map_df['replies_from'] = [[] for _ in range(len(author_map_df))]
    
    # iterate through authors and fill from dictionary
    for index, row in author_map_df.iterrows():
        author = row['author']
        replies_from = replies_from_dict[author]
        author_map_df.at[index, 'replies_from'] = replies_from
         
    return author_map_df[['author', 'replies_to', 'replies_from']]

# function to reorganize the reddit extraction data
def organize_reddit_content(reddit_extraction):
    '''
    Organized Structure:
        - url
        - title
        - subreddit
        - author
        - original_author (boolean for if they started the initial/original post)
        - author_upvotes (list)
        - author_dates (list)
        - replies_to (list of authors the author has responded to - can contain duplicates)
        - replies_from (list of authors that have responded to the author - can contain duplicates)
        - author_content (list of submissions - string values)
        - author_content_aggregated (single string value for all author's posts, comments, and replies on main post)
        
    Notes:
        - "replying_to" column in original extraction structure:
            - Root: original post
            - t3: comment under original post
            - t1: reply under comment
    '''
    
    # create copy
    cleaned_extraction = reddit_extraction.copy()
    
    # drop rows with Null content (likely link to outside source)
    cleaned_extraction.dropna(inplace=True, subset='content')
    cleaned_extraction.reset_index(inplace=True, drop=True)
    
    # remove duplicate posts
    cleaned_extraction.drop_duplicates(subset=['url', 'author', 'id', 'content', 'replying_to'], inplace=True, ignore_index=True)
    
    # each main post will have a main url
    urls = cleaned_extraction['url'].unique().tolist()
    
    # initialize dataframe
    reddit_organized_df = pd.DataFrame()
    
    # iterate through urls
    for url in urls:
        # create reorganized data structure for storage
        reddit_organized = {'url': [],
                            'title': [],
                            'subreddit': [],
                            'author': [],
                            'original_author': [],
                            'author_upvotes': [],
                            'author_dates': [],
                            'author_content': [],
                            'author_content_aggregated': []}
        
        # get reddit post
        reddit_post = cleaned_extraction[cleaned_extraction['url']==url]
        
        # create author map
        author_map = create_author_map(reddit_post)
        
        # get post title
        title = reddit_post['title'].unique()[0]
        
        # get subreddit
        subreddit = reddit_post['subreddit'].unique()[0]
        
        # get authors
        authors = reddit_post['author'].unique().tolist()
        
        # iterate through authors
        for author in authors:
            # subset reddit post on author
            author_subset = reddit_post[reddit_post['author']==author]
            
            # check for original author
            if 'Root' in author_subset['replying_to'].tolist():
                original_author = True
            else:
                original_author = False
            
            # author upvotes
            author_upvotes = author_subset['upvotes'].tolist()
            
            # author dates
            author_dates = author_subset['comment_date'].tolist()
            
            # get author content
            author_content = author_subset['content'].tolist()
            
            # aggregate author content
            author_content_aggregated = ' '.join(author_content)
            
            # populate data structure
            reddit_organized['url'].append(url)
            reddit_organized['title'].append(title)
            reddit_organized['subreddit'].append(subreddit)
            reddit_organized['author'].append(author)
            reddit_organized['original_author'].append(original_author)
            reddit_organized['author_upvotes'].append(author_upvotes)
            reddit_organized['author_dates'].append(author_dates)
            reddit_organized['author_content'].append(author_content)
            reddit_organized['author_content_aggregated'].append(author_content_aggregated)
            
        # create dataframe
        reddit_post_df = pd.DataFrame(reddit_organized)
        
        # merge in author mapping
        reddit_post_df = pd.merge(reddit_post_df, author_map, on='author')
        
        # concatenate with other url reorganizations
        reddit_organized_df = pd.concat([reddit_organized_df, reddit_post_df], ignore_index=True)

    return reddit_organized_df
    
'''
Corpus Process:
    1. folder name: search query
    2. file name:
        - root_id
        - type: root or comment
        - id
        - level: root, t3_{replying_to}, or t1_{replying_to}
    3. file content: content
'''

# function to allow permissions to remove folder
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# function to create corpus - creates a new folder
def create_reddit_corpus(reddit_query, cleaned_extraction, location='reddit_data_cleaned/corpora', replace_folder=True):
    # create corpus folder - account for if folder exists
    path = os.path.join(location, reddit_query)
    if (os.path.exists(path)) and (replace_folder):
        shutil.rmtree(path, onerror=remove_readonly)
        os.makedirs(path)
        print('Folder Replaced')
    elif (os.path.exists(path)) and not (replace_folder):
        print('Folder Already Exists, and Replacement Marked as False... Exiting Procedure')
        return
    else:
        os.makedirs(path)
        print('Folder Created')
        
    # get root ids
    root_ids = cleaned_extraction[cleaned_extraction['replying_to']=='Root'][['url', 'id']]
    
    # iterate through cleaned_extraction dataframe
    for index, row in cleaned_extraction.iterrows():
        # get replying_to value
        replying_to = row['replying_to']
        
        # root id
        root_id = root_ids[root_ids['url']==row['url']]['id'].iloc[0]
        
        # either root or content
        if replying_to == 'Root':
            filename_type = 'root'
            filename_level = 'root'
        else:
            filename_type = 'comment'
            filename_level = replying_to
        
        # id of post or comment
        filename_id = row['id']
        
        # file content
        file_content = row['content']
        
        # author
        author = row['author']
        
        # create text files - if not str then outside link posting
        if type(file_content) == str:
            filename = f'{root_id}_{filename_type}_{filename_id}_{filename_level}_{author}.txt'
            filepath = os.path.join(path, filename)
            with open(filepath, 'w') as file:
                try:
                    file.write(row['content'])
                except:
                    print(f"{row['content']}\n")
        else:
            continue
