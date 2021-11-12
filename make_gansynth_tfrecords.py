# modified make_tfrecords.py file from: https://github.com/skmhrk1209/GANSynth/blob/master/make_tfrecord.py
# uses the GANSynth (https://openreview.net/forum?id=H1xQVn09FX) filtering for NSynth
# shuffle, then split, to avoid separation by instrument-type
# note: modified to apply a custom split to include validation set in tfrecord
# -- need to use exact split for full comparison to original metrics

# to get data format, run:
# mkdir nsynth; cd nsynth
# wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
# wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
# wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz
# tar -xvf nsynth-train.jsonwav.tar.gz
# tar -xvf nsynth-valid.jsonwav.tar.gz
# tar -xvf nsynth-test.jsonwav.tar.gz
# rm nsynth-train.jsonwav.tar.gz nsynth-valid.jsonwav.tar.gz nsynth-test.jsonwav.tar.gz
# python make_gansynth_tfrecords.py

import tensorflow as tf
import pathlib
import random
import json

if __name__ == "__main__":

    random.seed(7)

    nsynth_all_examples = {}
    for filename in pathlib.Path("/data/vision/billf/scratch/yilundu/dataset/nsynth").glob("nsynth*/*.json"):
        with open(filename) as file:
            nsynth_examples = json.load(file)
            for key, value in nsynth_examples.items():
                value.update(dict(path=str(filename.parent/"audio"/f"{key}.wav")))
            nsynth_all_examples.update(nsynth_examples)

    nsynth_all_examples = list(nsynth_all_examples.items())
    random.shuffle(nsynth_all_examples)

    nsynth_all_examples_filter = []
    for i in range(100000):
        if nsynth_all_examples[i][1]['pitch'] < 100:
            nsynth_all_examples_filter.append(nsynth_all_examples[i])

    nsynth_all_examples = nsynth_all_examples_filter

    # train_size = 0.7
    # val_size = 0.1
    # nsynth_train_examples = nsynth_all_examples[:int(len(nsynth_all_examples) * train_size)]
    # nsynth_val_examples = nsynth_all_examples[int(len(nsynth_all_examples) * train_size):int(len(nsynth_all_examples) * (train_size + val_size))]
    # nsynth_test_examples = nsynth_all_examples[int(len(nsynth_all_examples) * (train_size + val_size)):]

    train_size = 10000
    val_size = 5000
    nsynth_train_examples = nsynth_all_examples[:train_size]
    nsynth_val_examples = nsynth_all_examples[train_size:train_size+val_size]
    nsynth_test_examples = nsynth_all_examples[train_size+val_size:train_size+(2*val_size)]

    json.dump(nsynth_train_examples, open("nsynth_train.json", "w"))
    json.dump(nsynth_val_examples, open("nsynth_valid.json", "w"))
    json.dump(nsynth_test_examples, open("nsynth_test.json", "w"))
    import pdb
    pdb.set_trace()
    print(nsynth_train_examples)
    print(nsynth_val_examples)
    print(nsynth_test_examples)

    # for nsynth_name, nsynth_examples in [("nsynth_train", nsynth_train_examples), ("nsynth_valid", nsynth_val_examples), ("nsynth_test", nsynth_test_examples)]:
    #     with tf.io.TFRecordWriter(f"/private/home/yilundu/sandbox/function-space-gan/data/nsynth/records/{nsynth_name}.tfrecord") as writer:
    #         for key, value in nsynth_examples:
    #             writer.write(
    #                 record=tf.train.Example(
    #                     features=tf.train.Features(
    #                         feature=dict(
    #                             path=tf.train.Feature(
    #                                 bytes_list=tf.train.BytesList(
    #                                     value=[value["path"].encode()]
    #                                 )
    #                             ),
    #                             pitch=tf.train.Feature(
    #                                 int64_list=tf.train.Int64List(
    #                                     value=[value["pitch"]]
    #                                 )
    #                             ),
    #                             source=tf.train.Feature(
    #                                 int64_list=tf.train.Int64List(
    #                                     value=[value["instrument_source"]]
    #                                 )
    #                             )
    #                         )
    #                     )
    #                 ).SerializeToString()
    #             )
