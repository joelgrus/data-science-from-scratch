
"""
This script walks us through the DataSciencester hypothetical example
"""

# list of users, each a dict of ID and name values
users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
]

# list of friendship pairs, each a tuple where the
# the 1st value represents a user and
# the 2nd value represents one friend of that user
friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# initialize the dict with an empty list for each user id
friendships = {user["id"]: [] for user in users}

# and loop over the freindship pairs to populate it
for user, friend in friendship_pairs:
    friendships[user].append(friend)
    friendships[friend].append(user)

def calculate_num_of_friends(user):
    """How many friends does _user_ have?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(calculate_num_of_friends(user) for user in users)

print(total_connections) # 24

# calculate average number of connections ----
# divide total_connections by number of users
num_users = len(users) # 10
avg_connections = total_connections / num_users # 24 / 10
print(avg_connections) # 2.4

# find most connected people ----

# create a list (user_id, number_of_friends)
num_friends_by_id = [(user["id"], calculate_num_of_friends(user))
                     for user in users]

print(f"Before sorting: {num_friends_by_id}")

# sort the list by number of friends in descending order
num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1],
                             reverse=True)

print(f"After sorting: {num_friends_by_id}")

# design a data scientists you may know feature ----
def foaf_ids_bad(user):
    """foaf is short for "friend of a friend" """
    return [foaf_id
            for friend_id in friendships[user["id"]]
            for foaf_id in friendships[friend_id]]

print(f"""
{foaf_ids_bad(users[0])}
It includes user 0 twice, since Hero is need friends with both of his friends.
It includes users 1 & 2, although they're already friends with Hero.
It includes user 3 twice since Chi is friends with Dunn and Sue.
""")

from collections import Counter

def friends_of_friends(user):
    """Count number of friends in common for friends of a friend"""
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]   # for each of my friends
        for foaf_id in friendships[friend_id]   # find their friends
        if foaf_id != user_id                   # who aren't me
        and foaf_id not in friendships[user_id] # and aren't my friends
    )

print(f"""
We see that Hero has two friends in common with Chi:
{friends_of_friends(users[0])}
""")

# users with similar interests ----
interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
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
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest):
    """Find the ids of all users who like the target interest"""
    return [user_id
            for user_id, user_interst in interests
            if user_interest == target_interst]

"""
This has to examine the whole list of interets for every search so its slow.
We're better off building an index from interests to users.
"""

from collections import defaultdict

# keys are interests, values are list of user_ids with that interest
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

print(user_ids_by_interest)

# add another from users to interests

# keys are user_ids, values are list of interests for that user_id
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

print(interests_by_user_id)

print("""
Find who has the most interests in common with a given users:
\t * Iterate over the user's interest
\t * For each interest, iterate over the other users with that interest
\t * Keep count of how many times we see each other user
""")

def most_common_interests_with(user):
    """Find who has most interests in common with _user_"""
    return Counter(
        interested_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"]
    )

print(most_common_interests_with(users[0]))

# salaries and experience ----
salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

import matplotlib.pyplot as plt

plt.scatter(x=[tenure[1] for tenure in salaries_and_tenures],
            y=[salary[0] for salary in salaries_and_tenures])
plt.xlabel("Years experience")
plt.ylabel("Salary (in 2019 dollars)")
plt.title("Salaries and Experience for Data Scientists, 2019")
plt.savefig("data_scientists_salary_tenure_scatter.png", dpi=300)

# keys are years, values are lists of the salaries for each tenure
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# keys are years, each value is average salary for that tenure
average_salary_by_tenure = {
    tenure: sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

print(average_salary_by_tenure)

def assign_tenure_bucket(tenure):
    """Bucket tenure into one of three groups"""
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"

# keys are tenure buckets, values are lists of salaries for that bucket
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = assign_tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# calculate average salary by tenure bucket
# keys are tenure buckets, values are average salary for that bucket
average_salary_by_bucket  = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}

print(average_salary_by_bucket)

# better understand topics of interest ----
print("""
One simple (if not particulary exciting) way to find the most popular
interests is to count the words:

\t 1. Lowercase each interest;
(since different users may or may not capitalize their interests)

\t 2. Split it into words; and

\t 3. Count the results.
""")

words_and_counts = Counter(word
                            for user, interest in interests
                            for word in interest.lower().split())

for word, count in words_and_counts.most_common():
    if count > 1:
        print(word, count)
