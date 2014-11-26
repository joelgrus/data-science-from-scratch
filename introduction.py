from __future__ import division

##########################
#                        #
# FINDING KEY CONNECTORS #
#                        #
##########################

users = [
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" },
    { "id": 10, "name": "Jen" } 
]

friendships = [(1,2),(1,3),(2,3),(2,4),(3,4),(4,5),
               (5,6),(6,7),(6,8),(7,9),(8,9),(9,10)]


users_by_id = { user["id"] : user 
                for user in users }

# first give each user an empty list
for user in users:
    user["friend_ids"] = []

# and then populate the lists with friendships    
for id1, id2 in friendships:
    users_by_id[id1]["friend_ids"].append(id2)
    users_by_id[id2]["friend_ids"].append(id1)    

total_connections = sum(len(user["friend_ids"]) for user in users)
num_users = len(users)
avg_connections = total_connections / num_users # 2.4

################################
#                              #
# DATA SCIENTISTS YOU MAY KNOW #
#                              #
################################

def friends_of_friends_bad(user):
    # "foaf" is short for "friend of a friend"
    return [foaf_id
            for friend_id in user["friend_ids"]
            for foaf_id in users_by_id[friend_id]["friend_ids"]]

def is_unknown_by(user, other_user_id):
    return (other_user_id != user["id"] and           # not me, and
            other_user_id not in user["friend_ids"])  # not my friend

def friends_of_friends(user):
    return Counter(foaf_id
        for friend_id in user["friend_ids"]             
        for foaf_id in users_by_id[friend_id]["friend_ids"]
        if is_unknown_by(user, foaf_id))

interests = [
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"), 
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"), 
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"), 
    (4, "machine learning"), (4, "regression"), (4, "decision trees"), 
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"), 
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"), 
    (6, "probability"), (6, "mathematics"), (6, "theory"), 
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"), 
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"), 
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"), 
    (9, "Java"), (9, "MapReduce"), (9, "Big Data"), (10, "statistics"), 
    (10, "R"), (10, "statsmodels")
]

def data_scientists_who_like(target_interest):
    return [user_id 
            for user_id, user_interest in interests 
            if user_interest == target_interest]

from collections import defaultdict

users_by_interest = defaultdict(list)

for user_id, interest in interests:
    users_by_interest[interest].append(user_id)

interests_by_user = defaultdict(list)

for user_id, interest in interests:
    interests_by_user[user_id].append(interest)

from collections import Counter

def most_common_interests_with(user_id):
    return Counter(interested_user_id
        for interest in interests_by_user["user_id"]   
        for interested_user_id in users_by_interest[interest]
        if interested_user_id != user_id)

###########################
#                         #
# SALARIES AND EXPERIENCE #
#                         #
###########################

salaries_and_tenures = [(83000, 8.7), (88000, 8.1), 
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

def make_chart_salaries_by_tenure(plt):
    tenures = [tenure for salary, tenure in salaries_and_tenures]
    salaries = [salary for salary, tenure in salaries_and_tenures]
    plt.scatter(tenures, salaries)
    plt.xlabel("Years Experience")
    plt.ylabel("Salary")
    plt.show()    

# keys are years
# values are the salaries for each tenure
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

average_salary_by_tenure = { 
    tenure : sum(salaries) / len(salaries)   
    for tenure, salaries in salary_by_tenure.items() 
}

def tenure_bucket(tenure):
    if tenure < 2: return "less than two"
    elif tenure < 5: return "between two and five"
    else: return "more than five"    

salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

average_salary_by_bucket = {
  tenure_bucket : sum(salaries) / len(salaries)
  for tenure_bucket, salaries in salary_by_tenure_bucket.iteritems()
}


#################
#               #
# PAID_ACCOUNTS #
#               #
#################

def predict_paid_or_unpaid(years_experience):
  if years_experience < 3.0: return "paid"
  elif years_experience < 8.5: return "unpaid"
  else: return "paid"

######################
#                    #
# TOPICS OF INTEREST #
#                    #
######################

words_and_counts = Counter(word
                           for user, interest in interests
                           for word in interest.lower().split())


if __name__ == "__main__":

    print
    print "######################"
    print "#"    
    print "# FINDING KEY CONNECTORS"
    print "#"
    print "######################"
    print


    print "total connections", total_connections
    print "number of users", num_users
    print "average connections", total_connections / num_users
    print

    print "users sorted by number of friends:"
    print sorted(users, 
                 key=lambda user: len(user["friend_ids"]), # by number of friends
                 reverse=True)                             # largest to smallest

    print
    print "######################"
    print "#"    
    print "# DATA SCIENTISTS YOU MAY KNOW"
    print "#"
    print "######################"
    print


    print "friends of friends bad for user 1:", friends_of_friends_bad(users_by_id[1])
    print "friends of friends for user 4:", friends_of_friends(users_by_id[4])

    print
    print "######################"
    print "#"    
    print "# SALARIES AND TENURES"
    print "#"
    print "######################"
    print

    print "average salary by tenure", average_salary_by_tenure
    print "average salary by tenure bucket", average_salary_by_bucket

    print
    print "######################"
    print "#"    
    print "# MOST COMMON WORDS"
    print "#"
    print "######################"
    print

    for word, count in words_and_counts.most_common():
        if count > 1:
            print word, count