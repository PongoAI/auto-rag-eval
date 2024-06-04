from openai import OpenAI
import pongo
from dotenv import load_dotenv
import os
import json
import math
load_dotenv()
together_api_key = os.environ.get("TOGETHER_API_KEY")
together_client = OpenAI(api_key=together_api_key, base_url='https://api.together.xyz/v1')

if os.environ.get("PONGO_API_KEY"):
    pongo_client = pongo.PongoClient(os.environ.get("PONGO_API_KEY"))
else:
    pongo_client = None




def _handle_assessment(query, docs, scoring_cutoff, run_only_pongo=False):
    output = []

    if(run_only_pongo):
        docs_for_pongo = []
        i = 0

        for doc in docs: #add ID field for pongo
            docs_for_pongo.append({'id': i, 'text': doc})
            i+=1

        pongo_response = pongo_client.filter(query=query, docs=docs_for_pongo, num_results=max(scoring_cutoff, len(docs_for_pongo)))
        docs = [pongo_doc['text'] for pongo_doc in pongo_response.json()]


    stringed_docs = []

    j = 0
    while j < len(docs) and j < scoring_cutoff:
        doc = docs[j]
        stringed_docs.append(f'\n-----------------\nSource #{j+1}:\n{doc}\n')
        j+=1
    
    k = 0
    while k < len(stringed_docs) and k < scoring_cutoff:
        curr_sources_string = '\n'.join(stringed_docs[k:k+5])
        k += 5

        llm_prompt = f'''**Task: Relevance Scoring**
    ================================

    You will be provided with a query and a list of sources and respond only with valid JSON thta can be parsed directly, never any other text. Your task is to read each source and determine how relevant it is to the query. Specifically, you will assign one of the following scores to each source:

    * **Highly Relevant (HR)**: The source provides direct, specific information that answers the query or is crucial to generating a response.
    * **Somewhat Relevant (SR)**: The source provides some useful information or context that could be used to generate a response, but is not directly answering the query.
    * **Not Relevant (NR)**: The source does not provide any useful information or context for generating a response to the query.

    Please score each source based on its relevance to the query.

    **Query:** {query}

    **Sources:** 
    {curr_sources_string}

    **Please respond ONLY with a JSON formatted list of scores, one for each source, in the format:**

    [
    {{"source_num": 1, "score": "HR"|"SR"|"NR"}},
    {{"source_num": 2, "score": "HR"|"SR"|"NR"}},
    {{"source_num": 3, "score": "HR"|"SR"|"NR"}},
    ...
    {{"source_num": n, "score": "HR"|"SR"|"NR"}}
    ]
        '''
        try:
            completion_response = json.loads(together_client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[{"role": "user", "content": llm_prompt}],
                stream=False,
                temperature=0.2,
            ).choices[0].message.content)
        except:
            print(f'LLM hallucinated, skipping this question: "{query}"')
            return {'status': 'error'}


        output.extend(completion_response)
    
    return {'scores': sorted(output, key=lambda x: x['source_num']), 'docs': stringed_docs, 'status': 'success'}


def _calculate_scores(scores, n):
    DCG = 0
    iDCG = 0
    hit_index = -1
    i = 1
    while i <= n and i < len(scores):
        curr_score = scores[i-1]

        iDCG += 1/math.log2(i+1)
        #add rating if relevant
        if curr_score['score'] == 'HR':
            DCG += 1/math.log2(i+1)

            if hit_index == -1:
                hit_index = i
        i+=1

        
    
    return {
        'DCG': DCG,
        'iDCG': iDCG,
        'nDCG': DCG/iDCG,
        'hit_index': hit_index,
        'did_hit': hit_index <= n and not hit_index == -1
    }

def run_assessment(query, docs, should_run_pongo, scoring_cutoff=5, results_filepath='./assessment.json', include_docs=True):
    """
    Run an assessment on the given query and documents, prints to the input file. Use evaluate_scores to do calculations across all assessments in a given output file.

    Parameters:
    query (str): The user query into a RAG (Retrieval-Augmented Generation) system.
    docs (list of str): A list of ordered strings from most relevant to least relevant to the query. Recommended to pass top 100-200 for Pongo
    scoring_cutoff (int): # of docs to include when calculating scores (MRR@X, NDCG@X, hit rate @ X). -1 includes all docs.
    should_run_pongo (boolean): Wether or not to run an assessment on potential gains from Pongo side-by-side. PONGO_API_KEY is requried in the .env if selected.
    results_filepath (string): The json to read assessments from
    save_docs (boolean): Wether or not to include the raw doc strings of ranked sources in output

    Returns:
    scores (obj)
    """

    output = {
        'status': 'success',
        'query': query
    }
    if should_run_pongo and not pongo_client:
        raise RuntimeError('Failed to create Pongo client, please make sure PONGO_API_KEY is defined in your .env')
    if not together_api_key:
        raise RuntimeError('TOGETHER_API_KEY not provided in .env')

    # handle file safety
    if os.path.exists(results_filepath):
        # If the file exists, check if it's empty
        with open(results_filepath, 'r+') as file:
            try:
                data = json.load(file)
                if data == "":  # If the file is empty
                    file.seek(0)
                    json.dump([], file)
                    file.truncate()
            except json.JSONDecodeError:
                # If the file is not a valid JSON, make it an empty array
                file.seek(0)
                json.dump([], file)
                file.truncate()
    else:
        # If the file does not exist, create it and make it an empty array
        with open(results_filepath, 'w') as file:
            json.dump([], file)

    # try:
    base_assessment = _handle_assessment(query, docs, scoring_cutoff)
    if base_assessment['status'] == 'error':
        return {'status': 'error'}

    base_calculated_scores = _calculate_scores(base_assessment['scores'], scoring_cutoff)
    output['base'] = {
        'docs': base_assessment['docs'],
        'relevance_ratings': base_assessment['scores'], 
        'calculated_scores': base_calculated_scores
    }
    if include_docs:
            output['base']['docs'] = base_assessment['docs']
    



    if should_run_pongo:
        pongo_assessment = _handle_assessment(query, docs, scoring_cutoff, run_only_pongo=True)
        if pongo_assessment['status'] == 'error':
            return {'status': 'error'}
        pongo_calculated_scores = _calculate_scores(pongo_assessment['scores'], scoring_cutoff)
        output['pongo'] = {
        'relevance_ratings': pongo_assessment['scores'], 
        'calculated_scores': pongo_calculated_scores
        }
        if include_docs:
            output['pongo']['docs'] = pongo_assessment['docs']
    

    with open(results_filepath, 'r+') as file:
        try:
            data = json.load(file)
            if isinstance(data, list):
                data.append(output)
            else:
                data = [output]
            file.seek(0)
            json.dump(data, file)
            file.truncate()
        except json.JSONDecodeError:
            # If the file is not a valid JSON, make it an array with the output
            file.seek(0)
            json.dump([output], file)
            file.truncate()

    # except:
    #     print(f'Failed to process: {query}')
    #     return {'status': 'error'}

def evaluate_scores(results_filepath='./assessment.json', scoring_cutoff=5):
    """
    Calculate scores for all assessments in a given output file

    Parameters:
    scoring_cutoff (int): # of docs to include when calculating scores (MRR@X, NDCG@X, hit rate @ X). -1 includes all docs.
    results_filepath (string): The json to read assessments from

    Returns:
    scores (obj)
    """
    pongo_ndcg = 0
    pongo_mrr = 0
    pongo_avg_hit_rate = 0

    base_ndcg = 0
    base_mrr = 0
    base_avg_hit_rate = 0



    with open(results_filepath, 'r') as file:
        data = json.load(file)
        num_elements = len(data)

        if data and len(data[0]['base']['relevance_ratings']) < scoring_cutoff:
            print(f"Warning: provided dataset was created with a scoring cutoff of {len(data['base']['relevance_ratings'])}, which is less than the evaluation cutoff provided ({scoring_cutoff}), so the lower value will be used in calculations.")
        i = 0
        for element in data:
            if element['status'] == 'error':
                print("Skipping error'ed element")
                continue

            base_scores =  _calculate_scores(element['base']['relevance_ratings'], scoring_cutoff) #re-calculate in case of smaller scoring_cutoff
            base_mrr += 1/base_scores['hit_index']
            base_ndcg += base_scores['nDCG']
            
            if base_scores['did_hit']:
                    base_avg_hit_rate += 1

            if 'pongo' in element:
                
                pongo_scores = _calculate_scores(element['pongo']['relevance_ratings'], scoring_cutoff) #re-calculate in case of smaller scoring_cutoff
                pongo_mrr += 1/pongo_scores['hit_index']
                if pongo_scores['did_hit']:
                    pongo_avg_hit_rate += 1


                pongo_ndcg += pongo_scores['nDCG']
            i+=1

        print('========== Base scores ==========')
        print(f'Base MRR @ {scoring_cutoff}: {base_mrr/num_elements}')
        print(f'Base hit rate @ {scoring_cutoff}: {base_avg_hit_rate/num_elements}')
        print(f'Base NDCG @ {scoring_cutoff}: {base_ndcg/num_elements}')

    if not pongo_avg_hit_rate == 0:
        print('\n\n========== Pongo scores ==========')
        print(f'Pongo MRR @ {scoring_cutoff}: {pongo_mrr/num_elements}')
        print(f'Pongo hit rate @ {scoring_cutoff}: {pongo_avg_hit_rate/num_elements}')
        print(f'Pongo NDCG @ {scoring_cutoff}: {pongo_ndcg/num_elements}')

        
            