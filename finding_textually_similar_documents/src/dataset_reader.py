import zipfile
from os import path

import json


def read_dataset(dataset_file, num_docs):

    docs = []
    # Define the relative path to the zipped data
    file_dir = path.dirname(__file__)   
    rel_path = f"../data/{dataset_file}"
    dataset_path = path.join(file_dir, rel_path)

    
    with zipfile.ZipFile(dataset_path, 'r') as zip:
        # find the name of the JSON file inside the zip file to open them 
        files_namelist = zip.namelist()[:num_docs]

        for file_name in files_namelist: 
            with zip.open(file_name) as file_json:
                file = json.load(file_json)

                # Retrieve the text of the articles.
                docs.append(file['text'])
    
    print(f'Data set {dataset_file} is processed. {num_docs} documents are saved.')
    return docs


if __name__ == '__main__':
    # zipped datasets: https://webz.io/free-datasets/technology-news-articles/
    dataset_file = 'tech_articles.zip' 
    num_docs = 5

    docs = read_dataset(dataset_file, num_docs)

