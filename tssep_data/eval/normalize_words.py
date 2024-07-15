"""


python -m tssep.eval.normalize_words c7_words_nemo_fix.json

"""
from tssep_data.io.json_results import load_json, dump_json


def main(*jsons):
    for json in jsons:
        data = load_json(json)
        entry: dict
        for entry in data:

            words = entry['words']
            entry.setdefault('words_orig', words)
            words = words.upper()
            # words = [word.replace(' ', '') for word in words]
            entry['words'] = words
        print(f'Write {json}')
        dump_json(data, json)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
