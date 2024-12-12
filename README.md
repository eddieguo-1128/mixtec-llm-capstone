# mixtec-llm-capstone

### Installation

```bash
conda create -n mixtec_llm python=3.11
conda activate mixtec_llm
pip install -r requirements.txt
```

### Few-shot prompting

`translate.py` is the main script to run LingoLLM. It has the following arguments:

- `--num_shots`: Number of few-shot examples to include in the prompt. Default is 3.
- `--metric`: Metric used for selecting few-shot examples. Options include `chrf` (default), `lexeme_recall`, or `random`.
- `--num_test`: Number of test sentences to translate from the test set. Default is 50.
- `--pipeline`: Pipeline configuration for the experiment. Options include:
  - `baseline`: Uses only few-shot examples.
  - `dict`: Includes a dictionary in the prompt along with few-shot examples.
  - `grammar`: Includes grammar rules in the prompt along with few-shot examples.
  - `dict_grammar`: Combines dictionary and grammar rules in the prompt along with few-shot examples.

Now let's see an example:

```bash
cd few_shot_prompting
python translate.py --num_shots 3 --metric "chrf" --num_test 50 --pipeline "dict"
```
