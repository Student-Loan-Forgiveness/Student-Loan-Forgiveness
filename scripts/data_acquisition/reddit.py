'''
Reddit Scraping and API
'''

# import libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import asyncpraw
import asyncio
import datetime
import os
from dotenv import load_dotenv

# reddit secrets
load_dotenv()
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT')
password = os.getenv('REDDIT_PASSWORD')
username = os.getenv('REDDIT_USERNAME')

# function to scrape reddit search query page
def search_reddit(query):
    '''
    Acts as if the user is typing a search query into reddit itself.

    Parameters
    ----------
    query : string
        Reddit search.

    Returns
    -------
    post_urls : list
        URLs from reddit search.

    '''
    # reddit base url
    url_base = 'https://www.reddit.com/search/?q='
    
    # format search query for url 
    url_query = query.lower().replace(' ', '+')
    
    # build full url
    url_built = f'{url_base}{url_query}'
    
    # return None if page cannot be scraped
    page = requests.get(url_built)
    if page.status_code != 200:
        print('Reddit Search Ineligible')
        return None
    soup = BeautifulSoup(page.text, 'lxml')
    posts = soup.find_all(class_='absolute inset-0')
    post_urls = [f'https://www.reddit.com{post["href"]}' for post in posts]
    
    return post_urls

# function to create content storage dictionary
def create_reddit_dictionary():
    '''
    Creates data structure to store Reddit post data in.

    Returns
    -------
    reddit_content_dict : dictionary
        Initialial data structure to store Reddit post data in.

    '''
    reddit_content_dict = {'url': [],
                           'title': [],
                           'original': [],
                           'self': [],
                           'post_date': [],
                           'comments': [],
                           'author': [],
                           'id': [],
                           'upvotes': [],
                           'content': [],
                           'comment_date': [],
                           'replying_to': [],
                           'subreddit': []}
    
    return reddit_content_dict

# function to populate content storage dictionary
def populate_reddit_dictionary(reddit_dict,
                               url,
                               title,
                               post_og,
                               post_self,
                               post_date,
                               comments,
                               author,
                               item_id,
                               upvotes,
                               content,
                               comment_date,
                               replying_to,
                               subreddit):
    '''
    Populate Reddit data structure.

    Parameters
    ----------
    reddit_dict : dictionary
        Iteratively populated Reddit data structure.
    url : string
        Reddit post url.
    title : string
        Reddit post title.
    post_og : integer
        Integer boolean for if original post.
    post_self : integer
        Integer boolean for if post is by author.
    post_date : string
        Date of post.
    comments : integer
        Number of comments.
    author : string
        Author of post, comment, or reply.
    item_id : string
        ID of post, comment, or reply.
    upvotes : integer
        Number of upvotes on post, comment, or reply.
    content : string
        Actual content of the post, comment, or reply. Can be NoneType if from outside source.
    comment_date : string
        Date of comment or reply.
    replying_to : string
        ID of post, comment, or reply this specific content is directly underneath.
    subreddit : string
        Name of Subreddit.

    Returns
    -------
    reddit_dict : dictionary
        Populated data strucutre of Reddit extraction.

    '''
    
    # populate dictionary
    reddit_dict['url'].append(url)
    reddit_dict['title'].append(title)
    reddit_dict['original'].append(post_og)
    reddit_dict['self'].append(post_self)
    reddit_dict['post_date'].append(post_date)
    reddit_dict['comments'].append(comments)
    reddit_dict['author'].append(author)
    reddit_dict['id'].append(item_id)
    reddit_dict['upvotes'].append(upvotes)
    reddit_dict['content'].append(content)
    reddit_dict['comment_date'].append(comment_date)
    reddit_dict['replying_to'].append(replying_to)
    reddit_dict['subreddit'].append(subreddit)
    
    return reddit_dict

# function to get reddit post content
def get_post_content(post_url, submission):
    '''
    Iterate through a single Reddit post for content in main, comments, and replies.

    Parameters
    ----------
    post_url : string
        Reddit post url.
    submission : Reddit API Object
        Reddit API Object containing the contents from a posting.

    Returns
    -------
    pandas dataframe
        Populated data strucutre of Reddit extraction in dataframe format.
    '''
    
    # generate content storage dictionary
    content_dict = create_reddit_dictionary()
    
    # post content
    post_title = submission.title
    post_id = submission.id
    post_og = submission.is_original_content # unique, own work
    post_self = submission.is_self # text only, no links to external websites
    post_date = datetime.datetime.utcfromtimestamp(submission.created_utc)
    post_author = submission.author.name
    post_upvote_num = submission.score
    post_content = submission.selftext
    post_comments = submission.num_comments
    post_subreddit = submission.subreddit.display_name
    
    # populate content dictionary
    content_dict = populate_reddit_dictionary(content_dict, post_url, post_title, post_og, post_self, post_date, post_comments, post_author, post_id, post_upvote_num, post_content, post_date, 'Root', post_subreddit)
    
    # debugging check
    print('Submission Extraction Successful\n')
    
    # post comments
    comments = submission.comments.list()
    for comment in comments:
        try:
            comment_author = comment.author.name
        except:
            continue
        comment_content = comment.body
        comment_date = datetime.datetime.utcfromtimestamp(comment.created_utc)
        comment_upvote_num = comment.score
        comment_id = comment.id
        replying_to = comment.parent_id
        replies = comment.replies.list()
        
        # populate content dictionary
        content_dict = populate_reddit_dictionary(content_dict, post_url, post_title, post_og, post_self, comment_date, len(replies), comment_author, comment_id, comment_upvote_num, comment_content, comment_date, replying_to, post_subreddit)
        
        # debugging check
        print('Comment Extraction Successful\n')
        
        # comment replies
        for reply in replies:
            try:
                reply_author = reply.author.name
            except:
                continue
            reply_content = reply.body
            reply_date = datetime.datetime.utcfromtimestamp(comment.created_utc)
            reply_upvote_num = reply.score
            reply_id = reply.id
            replying_to = reply.parent_id
            replies = reply.replies.list()
            
            # populate content dictionary
            content_dict = populate_reddit_dictionary(content_dict, post_url, post_title, post_og, post_self, reply_date, len(replies), reply_author, reply_id, reply_upvote_num, reply_content, reply_date, replying_to, post_subreddit)
    
            # debugging check
            print('Reply Extraction Successful\n')
    
    return pd.DataFrame(content_dict)

# function to ensure no duplicates
def eliminate_duplicates(query, reddit_data):
    '''
    Use for making additional Reddit searches.

    Parameters
    ----------
    query : string
        Reddit search.
    reddit_data : pandas dataframe
        Populated extracted Reddit data.

    Returns
    -------
    new_urls : list
        URLs from reddit search, with duplicates eliminated.

    '''
    # get current urls in data
    current_urls = reddit_data['url'].unique().tolist()
    
    # perform new search - search_reddit(query) returns urls
    new_search = search_reddit(query)
    
    # compare to get new urls
    new_urls = list(set(new_search) - set(current_urls))
    
    return new_urls

'''
Initial Data Retrieval
'''

# reddit search - web scraping
reddit_posts = search_reddit('student loan forgiveness')

# number of reddit posts with search query
total_posts = len(reddit_posts)

# reddit instantiation
reddit = asyncpraw.Reddit(client_id=client_id,
                          client_secret=client_secret,
                          user_agent=user_agent,
                          password=password,
                          username=username)

# initialize reddit content storage dictionary
reddit_content = pd.DataFrame(create_reddit_dictionary())

# iterate through the posts, unfortunately doesn't seem to operate properly within a function scope
for post_num, post in enumerate(reddit_posts):
    try:
        # call reddit post
        submission = await reddit.submission(url=post)
        
        # get reddit post content
        submission_content = get_post_content(post, submission)
        
        # concatenate post content with other posts' content
        reddit_content = pd.concat([reddit_content, submission_content], ignore_index=True)
    except:
        # error message
        print(f'Error on Post Number {post_num}.\n')
        continue
    
    # progress report
    progress = ((post_num + 1) / total_posts) * 100
    print(f'{progress:.2f}%')
    
# save dataframe
reddit_content.to_csv('reddit_data/student_loan_forgiveness_1_18_25.csv', index=False)

'''
Additional Data Retrievals - Round 1

- query: 'student loan forgiveness'
'''
# load reddit data
reddit_content = pd.read_csv('reddit_data/student_loan_forgiveness_1_18_25.csv')

# current urls (function takes this into account, but desired for further inspection)
current_urls = reddit_content['url'].unique().tolist()

# new search - same query
reddit_posts_updated = eliminate_duplicates('student loan forgiveness', reddit_content)

# number of reddit posts with search query
total_posts = len(reddit_posts_updated)

# initialize reddit content storage dictionary
reddit_content_updated = pd.DataFrame(create_reddit_dictionary())

# reddit instantiation
reddit = asyncpraw.Reddit(client_id=client_id,
                          client_secret=client_secret,
                          user_agent=user_agent,
                          password=password,
                          username=username)

# iterate through the posts, unfortunately doesn't seem to operate properly within a function scope
# change variable in the enumerate() call
# change variable involved in the dataframe concatenation
for post_num, post in enumerate(reddit_posts_updated):
    try:
        # call reddit post
        submission = await reddit.submission(url=post)
        
        # get reddit post content
        submission_content = get_post_content(post, submission)
        
        # concatenate post content with other posts' content
        reddit_content_updated = pd.concat([reddit_content_updated, submission_content], ignore_index=True)
    except:
        # error message
        print(f'Error on Post Number {post_num}.\n')
        continue
    
    # progress report
    progress = ((post_num + 1) / total_posts) * 100
    print(f'{progress:.2f}%')

# concatenate to original dataframe
reddit_content_updated = pd.concat([reddit_content, reddit_content_updated], ignore_index=True)

# save dataframe
reddit_content_updated.to_csv('reddit_data/student_loan_forgiveness_1_20_25.csv', index=False)

'''
Additional Data Retrievals - Round 2

- query: 'student loans' (general search)
'''
# load reddit data
reddit_content = pd.read_csv('reddit_data/student_loan_forgiveness_1_20_25.csv')

# current urls (function takes this into account, but desired for further inspection)
current_urls = reddit_content['url'].unique().tolist()

# new search - same query
reddit_posts_updated = eliminate_duplicates('student loans', reddit_content)

# number of reddit posts with search query
total_posts = len(reddit_posts_updated)

# initialize reddit content storage dictionary
reddit_content_updated = pd.DataFrame(create_reddit_dictionary())

# reddit instantiation
reddit = asyncpraw.Reddit(client_id=client_id,
                          client_secret=client_secret,
                          user_agent=user_agent,
                          password=password,
                          username=username)

# iterate through the posts, unfortunately doesn't seem to operate properly within a function scope
# change variable in the enumerate() call
# change variable involved in the dataframe concatenation
for post_num, post in enumerate(reddit_posts_updated):
    try:
        # call reddit post
        submission = await reddit.submission(url=post)
        
        # get reddit post content
        submission_content = get_post_content(post, submission)
        
        # concatenate post content with other posts' content
        reddit_content_updated = pd.concat([reddit_content_updated, submission_content], ignore_index=True)
    except:
        # error message
        print(f'Error on Post Number {post_num}.\n')
        continue
    
    # progress report
    progress = ((post_num + 1) / total_posts) * 100
    print(f'{progress:.2f}%')

# concatenate to original dataframe - new search query - don't update
# reddit_content_updated = pd.concat([reddit_content, reddit_content_updated], ignore_index=True)

# save dataframe
reddit_content_updated.to_csv('reddit_data/student_loans_1_20_25.csv', index=False)

'''
Additional Data Retrievals - Round 3

- query: 'is a college degree worth it' (general search)
'''
# load reddit data
reddit_content_set1 = pd.read_csv('reddit_data/student_loan_forgiveness_1_20_25.csv')
reddit_content_set2 = pd.read_csv('reddit_data/student_loans_1_20_25.csv')
reddit_content = pd.concat([reddit_content_set1, reddit_content_set2], ignore_index=True)

# current urls (function takes this into account, but desired for further inspection)
current_urls = reddit_content['url'].unique().tolist()

# new search - same query
reddit_posts_updated = eliminate_duplicates('is a college degree worth it', reddit_content)

# number of reddit posts with search query
total_posts = len(reddit_posts_updated)

# initialize reddit content storage dictionary
reddit_content_updated = pd.DataFrame(create_reddit_dictionary())

# reddit instantiation
reddit = asyncpraw.Reddit(client_id=client_id,
                          client_secret=client_secret,
                          user_agent=user_agent,
                          password=password,
                          username=username)

# iterate through the posts, unfortunately doesn't seem to operate properly within a function scope
# change variable in the enumerate() call
# change variable involved in the dataframe concatenation
for post_num, post in enumerate(reddit_posts_updated):
    try:
        # call reddit post
        submission = await reddit.submission(url=post)
        
        # get reddit post content
        submission_content = get_post_content(post, submission)
        
        # concatenate post content with other posts' content
        reddit_content_updated = pd.concat([reddit_content_updated, submission_content], ignore_index=True)
    except:
        # error message
        print(f'Error on Post Number {post_num}.\n')
        continue
    
    # progress report
    progress = ((post_num + 1) / total_posts) * 100
    print(f'{progress:.2f}%')

# concatenate to original dataframe - new search query - don't update
# reddit_content_updated = pd.concat([reddit_content, reddit_content_updated], ignore_index=True)

# save dataframe
reddit_content_updated.to_csv('reddit_data/degree_worth_1_20_25.csv', index=False)
