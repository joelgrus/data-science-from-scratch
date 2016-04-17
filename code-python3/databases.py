import math, random, re
from collections import defaultdict

class Table:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def __repr__(self):
        """pretty representation of the table: columns then rows"""
        return str(self.columns) + "\n" + "\n".join(map(str, self.rows))

    def insert(self, row_values):
        if len(row_values) != len(self.columns):
            raise TypeError("wrong number of elements")
        row_dict = dict(zip(self.columns, row_values))
        self.rows.append(row_dict)

    def update(self, updates, predicate):
        for row in self.rows:
            if predicate(row):
                for column, new_value in updates.items():
                    row[column] = new_value

    def delete(self, predicate=lambda row: True):
        """delete all rows matching predicate
        or all rows if no predicate supplied"""
        self.rows = [row for row in self.rows if not(predicate(row))]

    def select(self, keep_columns=None, additional_columns=None):

        if keep_columns is None:         # if no columns specified,
            keep_columns = self.columns  # return all columns

        if additional_columns is None:
            additional_columns = {}

        # new table for results
        result_table = Table(keep_columns + list(additional_columns.keys()))

        for row in self.rows:
            new_row = [row[column] for column in keep_columns]
            for column_name, calculation in additional_columns.items():
                new_row.append(calculation(row))
            result_table.insert(new_row)

        return result_table

    def where(self, predicate=lambda row: True):
        """return only the rows that satisfy the supplied predicate"""
        where_table = Table(self.columns)
        where_table.rows = list(filter(predicate, self.rows))
        return where_table

    def limit(self, num_rows=None):
        """return only the first num_rows rows"""
        limit_table = Table(self.columns)
        limit_table.rows = (self.rows[:num_rows]
                            if num_rows is not None
                            else self.rows)
        return limit_table

    def group_by(self, group_by_columns, aggregates, having=None):

        grouped_rows = defaultdict(list)

        # populate groups
        for row in self.rows:
            key = tuple(row[column] for column in group_by_columns)
            grouped_rows[key].append(row)

        result_table = Table(group_by_columns + list(aggregates.keys()))

        for key, rows in grouped_rows.items():
            if having is None or having(rows):
                new_row = list(key)
                for aggregate_name, aggregate_fn in aggregates.items():
                    new_row.append(aggregate_fn(rows))
                result_table.insert(new_row)

        return result_table

    def order_by(self, order):
        new_table = self.select()       # make a copy
        new_table.rows.sort(key=order)
        return new_table

    def join(self, other_table, left_join=False):

        join_on_columns = [c for c in self.columns           # columns in
                           if c in other_table.columns]      # both tables

        additional_columns = [c for c in other_table.columns # columns only
                              if c not in join_on_columns]   # in right table

        # all columns from left table + additional_columns from right table
        join_table = Table(self.columns + additional_columns)

        for row in self.rows:
            def is_join(other_row):
                return all(other_row[c] == row[c] for c in join_on_columns)

            other_rows = other_table.where(is_join).rows

            # each other row that matches this one produces a result row
            for other_row in other_rows:
                join_table.insert([row[c] for c in self.columns] +
                                  [other_row[c] for c in additional_columns])

            # if no rows match and it's a left join, output with Nones
            if left_join and not other_rows:
                join_table.insert([row[c] for c in self.columns] +
                                  [None for c in additional_columns])

        return join_table

if __name__ == "__main__":

    users = Table(["user_id", "name", "num_friends"])
    users.insert([0, "Hero", 0])
    users.insert([1, "Dunn", 2])
    users.insert([2, "Sue", 3])
    users.insert([3, "Chi", 3])
    users.insert([4, "Thor", 3])
    users.insert([5, "Clive", 2])
    users.insert([6, "Hicks", 3])
    users.insert([7, "Devin", 2])
    users.insert([8, "Kate", 2])
    users.insert([9, "Klein", 3])
    users.insert([10, "Jen", 1])

    print("users table")
    print(users)
    print()

    # SELECT

    print("users.select()")
    print(users.select())
    print()

    print("users.limit(2)")
    print(users.limit(2))
    print()

    print("users.select(keep_columns=[\"user_id\"])")
    print(users.select(keep_columns=["user_id"]))
    print()

    print('where(lambda row: row["name"] == "Dunn")')
    print(users.where(lambda row: row["name"] == "Dunn")
               .select(keep_columns=["user_id"]))
    print()

    def name_len(row): return len(row["name"])

    print('with name_length:')
    print(users.select(keep_columns=[],
                       additional_columns = { "name_length" : name_len }))
    print()

    # GROUP BY

    def min_user_id(rows): return min(row["user_id"] for row in rows)

    stats_by_length = users \
        .select(additional_columns={"name_len" : name_len}) \
        .group_by(group_by_columns=["name_len"],
                  aggregates={ "min_user_id" : min_user_id,
                               "num_users" : len })

    print("stats by length")
    print(stats_by_length)
    print()

    def first_letter_of_name(row):
        return row["name"][0] if row["name"] else ""

    def average_num_friends(rows):
        return sum(row["num_friends"] for row in rows) / len(rows)

    def enough_friends(rows):
        return average_num_friends(rows) > 1

    avg_friends_by_letter = users \
        .select(additional_columns={'first_letter' : first_letter_of_name}) \
        .group_by(group_by_columns=['first_letter'],
                  aggregates={ "avg_num_friends" : average_num_friends },
                  having=enough_friends)

    print("avg friends by letter")
    print(avg_friends_by_letter)
    print()

    def sum_user_ids(rows): return sum(row["user_id"] for row in rows)

    user_id_sum = users \
        .where(lambda row: row["user_id"] > 1) \
        .group_by(group_by_columns=[],
                  aggregates={ "user_id_sum" : sum_user_ids })

    print("user id sum")
    print(user_id_sum)
    print()

    # ORDER BY

    friendliest_letters = avg_friends_by_letter \
        .order_by(lambda row: -row["avg_num_friends"]) \
        .limit(4)

    print("friendliest letters")
    print(friendliest_letters)
    print()

    # JOINs

    user_interests = Table(["user_id", "interest"])
    user_interests.insert([0, "SQL"])
    user_interests.insert([0, "NoSQL"])
    user_interests.insert([2, "SQL"])
    user_interests.insert([2, "MySQL"])

    sql_users = users \
    .join(user_interests) \
    .where(lambda row: row["interest"] == "SQL") \
    .select(keep_columns=["name"])

    print("sql users")
    print(sql_users)
    print()

    def count_interests(rows):
        """counts how many rows have non-None interests"""
        return len([row for row in rows if row["interest"] is not None])

    user_interest_counts = users \
        .join(user_interests, left_join=True) \
        .group_by(group_by_columns=["user_id"],
                  aggregates={"num_interests" : count_interests })

    print("user interest counts")
    print(user_interest_counts)

    # SUBQUERIES

    likes_sql_user_ids = user_interests \
        .where(lambda row: row["interest"] == "SQL") \
        .select(keep_columns=['user_id'])

    likes_sql_user_ids.group_by(group_by_columns=[],
                                aggregates={ "min_user_id" : min_user_id })

    print("likes sql user ids")
    print(likes_sql_user_ids)
