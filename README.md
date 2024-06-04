# Automated RAG Retrieval Eval

Automate retrieval result relevance scoring using Llama3 70B in one line of code, optionally compare your existing results to what they could be with Pongo's semantic filter on MRR@X, hit rate @ X, and NDCG@X (default x=5). 

Try ```example.py``` to see it in action

## How it works
1. Drop ```assess.py``` into your existing testbench workspace.

2. Add the API keys (see next section) to your .env

3. On each query in your existing eval suite, pass in the top 100-200 results and query string to ```run_assessment()``` in ```assess.py```.  This will run the LLM assessment on that given datapoint and store the results in a JSON.

4. In the end, run ```evaluate_scores()``` to calculate accuracy metrics.

## API keys
Make sure you add a .env with ```TOGETHER_API_KEY``` (and [here](https://together.ai/)) for the LLM, or feel free to swap out to your preferred LLM provider. Ff you're assessing pongo as well, a ```PONGO_API_KEY``` (get one [here](https://joinpongo.com/)).

## Accuracy
~90-95% as accurate as humans on most questions, performs especially well on direct fact-based questions, is generally more lenient than a human on open-ended questions, and gets tripped on up multi-hop reasoning.
