import praw
import csv
from transformers import pipeline


reddit = praw.Reddit(
    client_id='g3Lyl8rNvnDlj9iuDyA5pA',           
    client_secret='Xg5hoxNxV-Ir9jZoX5k9d_zzgctdpg',   
    user_agent='python:nasa_merch_scraper:v1.0 (by u/False-Interaction877)',        
)


post_urls = [
    'https://www.reddit.com/r/OutOfTheLoop/comments/95pfzd/why_are_nasa_tshirts_so_popular_as_of_late/',
    'https://www.reddit.com/r/college/comments/9bvirm/why_do_so_many_students_wear_nasa_shirts/',
    'https://www.reddit.com/r/NoStupidQuestions/comments/9tk0jw/why_is_clothing_with_the_nasa_logo_suddenly_so/',
    'https://www.reddit.com/r/NoStupidQuestions/comments/1ai0nn0/why_are_nasa_shirts_so_popular/',
    'https://www.reddit.com/r/answers/comments/dh72g0/why_are_so_many_young_people_wearing_nasa_branded/',
    'https://www.reddit.com/r/OutOfTheLoop/comments/apcckd/whats_up_with_everyone_wearing_nasa_merch/',
    'https://www.reddit.com/r/germany/comments/16uh2aq/is_it_me_or_nasa_is_considered_just_another/',
    'https://www.reddit.com/r/ask/comments/o1b7nn/why_do_so_many_un%D0%B5du%D1%81%D0%B0t%D0%B5d_people_wear/',
    'https://www.reddit.com/r/NoStupidQuestions/comments/zvtwus/why_are_people_wearing_all_this_nasa_clothing/',
    'https://www.reddit.com/r/TooAfraidToAsk/comments/9q4l0o/why_is_everyone_wearing_nasa_logo_clothing_now/',
    'https://www.reddit.com/r/unpopularopinion/comments/b5xmqp/wearing_nasa_merchandise_doesnt_make_you_smart_or/',
    'https://www.reddit.com/r/AskReddit/comments/k62625/people_who_buy_nasa_shirts_at_popular_clothing/',
]


comments = []

for post_url in post_urls:
    submission = reddit.submission(url=post_url)
    submission.comments.replace_more(limit=0)  
    for comment in submission.comments.list():
        comments.append(comment.body)


print(comments)


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Interest in space and NASA's work", "Trending or Cool Clothing/Logo", "Symbol of US Accomplishments in Space Discovery/Patriotism", "Cheap and Conveniently Located Clothing Stores", "Other"]
for comment in comments:
    result = classifier(comment, candidate_labels)
    print(f"Comment: {comment}")
    print(f"Classified as: {result['labels'][0]} with confidence {result['scores'][0]:.4f}\n")



