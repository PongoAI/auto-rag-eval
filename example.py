from assess import run_assessment, evaluate_scores

quereis = ['Is water wet?', 'What is OpenAI?', 'What colors can apples be?'] 

for query in quereis:

    #do your exisitng pipeline search here, 
    results = ['Apples are red or green.', 'Openai is a software company', 'Water is wet'] * 100 #total of 300 docs being passed in

    #Pass list of strings + results to eval. If running pongo, pass the top 100-200 for it to filter
    #Make sure TOGETHER_API_KEY and PONGO_API_KEY are defined
    run_assessment(query=query, docs=results, should_run_pongo=True, scoring_cutoff=5, include_docs=False, results_filepath='./assessment-results.json')

#reads assessed scored from the assessment.json file written by run_assessment, print out results
evaluate_scores(results_filepath='./assessment-results.json', scoring_cutoff=5)