import requests
import json

def request_api_codeforces(method_name):
    address = 'https://codeforces.com/api/{methodName}'
    address = address.format(methodName=method_name)
    response = requests.get(address)
    obj = response.json()
    print(method_name, obj['status'])

    return obj['result']

with open("data/contests.json", "w") as outfile:
    json.dump(request_api_codeforces('contest.list'), outfile)

with open("data/problems.json", "w") as outfile:
    json.dump(request_api_codeforces('problemset.problems'), outfile)

tags_dataset = dict()
with open('data/problems.json') as f:
    data = json.load(f)
    problems = data['problems']
    for problem in problems:
        problem_id = str(problem['contestId']) + problem['index']
        tags = problem['tags']
        tags_dataset[problem_id] = tags

with open("data/tags.json", "w") as outfile:
    json.dump(tags_dataset, outfile)