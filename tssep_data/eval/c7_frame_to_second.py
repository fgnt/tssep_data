"""


python -m tssep.eval.c7_frame_to_second

"""

import decimal
from pathlib import Path
import paderbox as pb
import padertorch as pt
import meeteval.io.chime7
from tssep_data.io.json_results import load_json, dump_json


class StartOffset:
    def __init__(self, ds):
        self.ds = ds
        self.cache = {}

    def __getitem__(self, session_id):
        if session_id not in self.cache:
            try:
                start_offset = self.ds[session_id].get('start', 0)
            except KeyError:
                start_offsets = {v.get('start', 0) for k, v in self.ds.items() if k.startswith(session_id)}
                assert len(start_offsets) == 1, (start_offsets, session_id, self.ds.keys())
                start_offset = start_offsets.pop()
            self.cache[session_id] = start_offset

        return self.cache[session_id]


def main(
        json,
        # json='c7_words_nemo.json',
        out='{json.parent}/{json.stem}_fix.json',
        config='config.yaml::eeg.probability_to_segments',
        dataset=None,
):
    # folder = Path(folder)
    json = Path(json)
    data = load_json(json)

    if all(['start_time' in entry for entry in data]):
        print(f'Nothing to do. All entries in {json} have already a start_time.')
    else:
        config_file, probability_to_segments_path = config.split('::')
        cfg = pb.io.load(config_file)

        reader = pt.configurable.Configurable.from_config(
            cfg['eeg']['reader'] or cfg['eg']['trainer']['model']['reader'])
        if dataset is None:
            dataset = reader.eval_dataset_name
        ds = reader(dataset, load_audio=False)
        pts = pt.configurable.Configurable.from_config(
            pb.utils.nested.get_by_path(cfg, probability_to_segments_path)
        )
        start_offset = StartOffset(ds)
        sample_rate = reader.sample_rate

        for entry in data:
            if 'start_time' not in entry:
                assert 'end_time' not in entry, entry
                session_id = entry['session_id']
                start, end = pts.frames_to_samples(entry['start_frame'], entry['stop_frame'], start_offset[session_id])

                entry['start_sample'] = start
                entry['end_sample'] = end
                entry['start_time'] = decimal.Decimal(start) / sample_rate
                entry['end_time'] = decimal.Decimal(end) / sample_rate

    out = Path(out.format(json=json))
    if out.exists():
        assert not out.samefile(json), (out, json)
    dump_json(data, out)
    print(f'Wrote {out}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
