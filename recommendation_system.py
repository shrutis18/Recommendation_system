import pyspark
import sys
from pyspark import SparkContext
sc= SparkContext("local", "bda")

import json
import numpy as np

filename = sys.argv[1]

jsons = sc.textFile(filename).map(lambda d: json.loads(d))

# asins = ['B00EZPXYP4', 'B00CTTEKJW']

asins = eval(sys.argv[2])

def compute_cosine_sim_num(intersection_users, target_item_user_ratings, comparable_item_list):
    cosine_sim_num = 0
    for user in intersection_users:
        cosine_sim_num += dict(target_item_user_ratings)[user] * dict(comparable_item_list)[user]
    return cosine_sim_num


def neighbours_with_min_similar_users(comparable_item_list):
    targets = target_items_shared.value
    res = []
    comparable_item_list_users = list(map(lambda m: m[0], comparable_item_list))
    mean_comparable_user_rating = np.mean(list(map(lambda m: m[1], comparable_item_list)))
    mean_center_comparable_list = list(map(lambda m: (m[0], m[1] - mean_comparable_user_rating), comparable_item_list))
    for (target_item, target_item_user_ratings) in targets:
        cosine_sim_num = 0
        user_rating_intersection = list(
            set(list(map(lambda m: m[0], target_item_user_ratings))).intersection(comparable_item_list_users))
        mean_target_user_rating = np.mean(list(map(lambda m: m[1], target_item_user_ratings)))
        mean_center_target_item_user_ratings = list(
            map(lambda m: (m[0], m[1] - mean_target_user_rating), target_item_user_ratings))
        if len(user_rating_intersection) > 1:
            cosine_sim_num = compute_cosine_sim_num(user_rating_intersection, mean_center_target_item_user_ratings,
                                                    mean_center_comparable_list)
            vector1 = list(map(lambda t: t[1], mean_center_target_item_user_ratings))
            vector2 = list(map(lambda t: t[1], mean_center_comparable_list))
            cosine_sim_den = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2)))
        if cosine_sim_num > 0 and cosine_sim_den > 0:
            res.append((cosine_sim_num / cosine_sim_den, target_item, comparable_item_list))
    return res


def compute_target_items_with_users_with_no_rating(target_item_list):
    unique_users = unique_users_shared.value
    users_in_target_item = list(map(lambda i: i[0], target_item_list))
    users_not_in_target_item = list(filter(lambda user: user not in users_in_target_item, unique_users))
    return users_not_in_target_item


def compute_uniq_target_unknown(item, users_list, neighbours):
    flat_target_user_list = []
    for user in users_list:
        flat_target_user_list.append((item, user, neighbours))
    return flat_target_user_list


def filter_zero_neighbours(unknown_user, neighbours):
    res = []
    for (sim, user_rating_kvs) in neighbours:
        user_rating_kvs_dict = dict(user_rating_kvs)
        if unknown_user in user_rating_kvs_dict:
            res.append((sim, user_rating_kvs_dict[unknown_user]))
    return res


def weighted_average(sim_rating_kvs):
    num = sum(map(lambda t: t[0] * t[1], sim_rating_kvs))
    den = sum(map(lambda t: t[0], sim_rating_kvs))
    return num / den


unique_users_with_item = jsons.map(lambda r: ((r['reviewerID'], r['asin']), r['overall'])).groupByKey().map(
    lambda t: (t[0], list(t[1])))
unique_users_with_recently_rated = unique_users_with_item.map(lambda v: (v[0], v[1][-1]))
record_grouped_by_items = unique_users_with_recently_rated.map(lambda r: (r[0][1], (r[0][0], r[1]))).groupByKey().map(
    lambda t: (t[0], list(t[1])))
record_grouped_by_items_filtered_25_users = record_grouped_by_items.filter(lambda f: len(f[1]) >= 25)
record_grouped_by_items_filtered_25_users_flat = record_grouped_by_items_filtered_25_users.flatMapValues(lambda t: t)
record_grouped_by_users = record_grouped_by_items_filtered_25_users_flat.map(
    lambda u: (u[1][0], (u[0], u[1][1]))).groupByKey().map(lambda t: (t[0], list(t[1])))
record_grouped_by_users_filtered_5_items = record_grouped_by_users.filter(lambda f: len(f[1]) >= 5)
record_grouped_by_users_filtered_5_items_flat = record_grouped_by_users_filtered_5_items.flatMapValues(lambda t: t)
preprocessed_records = record_grouped_by_users_filtered_5_items_flat.map(
    lambda i: (i[1][0], (i[0], i[1][1]))).groupByKey().map(lambda t: (t[0], list(t[1])))

unique_users = record_grouped_by_users_filtered_5_items.map(lambda u: u[0]).take(1000)
unique_users_shared = sc.broadcast(unique_users)

target_items = preprocessed_records.filter(lambda rec: rec[0] in asins)
target_items_shared = sc.broadcast(target_items.take(1000))

neighbours_grouped_by_target_asins = preprocessed_records.flatMap(lambda t: neighbours_with_min_similar_users(t[1])).map(
    lambda t: (t[1], (t[0], t[2]))).groupByKey().map(lambda t: (t[0], list(t[1])))

target_items_with_unknown_user_ratings = target_items.map(
    lambda t: (t[0], compute_target_items_with_users_with_no_rating(t[1])))
target_items_with_unknown_user_with_neighbours = target_items_with_unknown_user_ratings.join(
    neighbours_grouped_by_target_asins)

target_items_with_unknown_user_with_neighbours_dist = target_items_with_unknown_user_with_neighbours.flatMap(
    lambda t: compute_uniq_target_unknown(t[0], t[1][0], t[1][1]))
target_items_with_unknown_user_with_neighbours_dist_filtered_two = target_items_with_unknown_user_with_neighbours_dist.map(
    lambda t: (t[0], t[1], filter_zero_neighbours(t[1], t[2]))).filter(lambda t: len(t[2]) > 1)
target_items_with_unknown_user_with_recommended_ratings = target_items_with_unknown_user_with_neighbours_dist_filtered_two.map(
    lambda t: (t[0], t[1], weighted_average(t[2])))

# unique_users_shared.value

print(target_items_with_unknown_user_with_recommended_ratings.take(1000))